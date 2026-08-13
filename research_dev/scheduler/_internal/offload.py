"""Composition of host work, phone operator work, and transport."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .operator_split import (
    OperatorSplitInvocation,
    OperatorSplitPolicy,
)
from .phone_transport import DmaBufTransfer, PhoneTransportContract


__all__ = [
    "OFFLOAD_CONTRACT_SCHEMA",
    "BackendArtifact",
    "OffloadError",
    "OperatorOffloadContract",
    "OperatorOffloadInvocation",
    "PhoneBackendConfig",
]


OFFLOAD_CONTRACT_SCHEMA = "research-scheduler-operator-offload-v1"


class OffloadError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise OffloadError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise OffloadError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise OffloadError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(name: str, value: object) -> str:
    result = _text(name, value).removeprefix("sha256:")
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise OffloadError(f"{name} must be a lowercase SHA-256")
    return "sha256:" + result


@dataclass(frozen=True)
class BackendArtifact:
    role: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _text("artifact role", self.role)
        if not _text("artifact path", self.path).startswith("/"):
            raise OffloadError("artifact path must be absolute")
        object.__setattr__(self, "sha256", _sha256("artifact sha256", self.sha256))

    @classmethod
    def from_json(cls, value: object) -> "BackendArtifact":
        if type(value) is not dict:
            raise OffloadError("backend artifact must be an object")
        return cls(
            role=value.get("role"),
            path=value.get("path"),
            sha256=value.get("sha256"),
        )

    def to_json(self) -> dict[str, str]:
        return {"path": self.path, "role": self.role, "sha256": self.sha256}


@dataclass(frozen=True)
class PhoneBackendConfig:
    backend_id: str
    phone_serial: str
    adb_port: int
    compute_backend: str
    layer_spec: str
    bridge_bind: str
    bridge_port: int
    server_timeout_ms: int
    session_timeout_s: int
    max_requests: int
    artifacts: tuple[BackendArtifact, ...]
    resident_weight_bytes: int | None = None
    resident_weight_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "backend_id",
            "phone_serial",
            "compute_backend",
            "layer_spec",
            "bridge_bind",
        ):
            _text(name, getattr(self, name))
        _integer("adb_port", self.adb_port, 1)
        _integer("bridge_port", self.bridge_port, 1)
        if self.bridge_port > 65535:
            raise OffloadError("bridge_port exceeds 65535")
        _integer("server_timeout_ms", self.server_timeout_ms, 1)
        _integer("session_timeout_s", self.session_timeout_s, 1)
        _integer("max_requests", self.max_requests, 1)
        artifacts = tuple(self.artifacts)
        if (
            not artifacts
            or any(not isinstance(item, BackendArtifact) for item in artifacts)
            or len({item.role for item in artifacts}) != len(artifacts)
        ):
            raise OffloadError("backend artifacts must be non-empty and unique")
        required = {
            "bridge",
            "cold_model",
            "phone_model",
            "phone_session",
            "phone_worker",
            "restore_usb",
        }
        if not required.issubset(item.role for item in artifacts):
            raise OffloadError("backend artifacts are incomplete")
        object.__setattr__(self, "artifacts", artifacts)
        if (self.resident_weight_bytes is None) != (
            self.resident_weight_budget_bytes is None
        ):
            raise OffloadError(
                "phone resident weight size and budget must appear together"
            )
        if self.resident_weight_bytes is not None:
            _integer(
                "phone resident_weight_bytes", self.resident_weight_bytes, 1
            )
            _integer(
                "phone resident_weight_budget_bytes",
                self.resident_weight_budget_bytes,
                1,
            )
            if self.resident_weight_bytes > self.resident_weight_budget_bytes:
                raise OffloadError("phone resident weights exceed memory budget")

    @classmethod
    def from_json(cls, value: object) -> "PhoneBackendConfig":
        if type(value) is not dict:
            raise OffloadError("phone backend must be an object")
        raw_artifacts = value.get("artifacts")
        if type(raw_artifacts) is not list:
            raise OffloadError("phone backend artifacts must be a list")
        return cls(
            backend_id=value.get("backend_id"),
            phone_serial=value.get("phone_serial"),
            adb_port=value.get("adb_port"),
            compute_backend=value.get("compute_backend"),
            layer_spec=value.get("layer_spec"),
            bridge_bind=value.get("bridge_bind"),
            bridge_port=value.get("bridge_port"),
            server_timeout_ms=value.get("server_timeout_ms"),
            session_timeout_s=value.get("session_timeout_s"),
            max_requests=value.get("max_requests"),
            artifacts=tuple(
                BackendArtifact.from_json(item) for item in raw_artifacts
            ),
            resident_weight_bytes=value.get("resident_weight_bytes"),
            resident_weight_budget_bytes=value.get(
                "resident_weight_budget_bytes"
            ),
        )

    def artifact(self, role: str) -> BackendArtifact:
        _text("artifact role", role)
        try:
            return next(item for item in self.artifacts if item.role == role)
        except StopIteration as exc:
            raise OffloadError(f"backend artifact is missing: {role}") from exc

    def to_json(self) -> dict[str, object]:
        value: dict[str, object] = {
            "adb_port": self.adb_port,
            "artifacts": [item.to_json() for item in self.artifacts],
            "backend_id": self.backend_id,
            "bridge_bind": self.bridge_bind,
            "bridge_port": self.bridge_port,
            "compute_backend": self.compute_backend,
            "layer_spec": self.layer_spec,
            "max_requests": self.max_requests,
            "phone_serial": self.phone_serial,
            "server_timeout_ms": self.server_timeout_ms,
            "session_timeout_s": self.session_timeout_s,
        }
        if self.resident_weight_bytes is not None:
            value["resident_weight_budget_bytes"] = (
                self.resident_weight_budget_bytes
            )
            value["resident_weight_bytes"] = self.resident_weight_bytes
        return value


@dataclass(frozen=True)
class OperatorOffloadInvocation:
    split: OperatorSplitInvocation
    transfer: DmaBufTransfer
    required_resource_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.split, OperatorSplitInvocation):
            raise OffloadError("offload split invocation is invalid")
        if not isinstance(self.transfer, DmaBufTransfer):
            raise OffloadError("offload transfer is invalid")
        resources = tuple(self.required_resource_ids)
        if not resources or len(resources) != len(set(resources)):
            raise OffloadError("offload resources must be non-empty and unique")
        object.__setattr__(self, "required_resource_ids", resources)


@dataclass(frozen=True)
class OperatorOffloadContract:
    route_id: str
    host_resource_id: str
    split: OperatorSplitPolicy
    transport: PhoneTransportContract
    backend: PhoneBackendConfig

    def __post_init__(self) -> None:
        _text("offload route_id", self.route_id)
        _text("offload host_resource_id", self.host_resource_id)
        if not isinstance(self.split, OperatorSplitPolicy):
            raise OffloadError("offload split policy is invalid")
        if not isinstance(self.transport, PhoneTransportContract):
            raise OffloadError("offload transport is invalid")
        if not isinstance(self.backend, PhoneBackendConfig):
            raise OffloadError("offload backend is invalid")
        if self.split.io_type != self.transport.io_type:
            raise OffloadError("offload split and transport I/O differ")
        if len(self.split.alternate_columns) > 1:
            raise OffloadError("phone backend supports one alternate column width")
        expected_payload = (
            self.split.n_embd
            * self.split.max_tokens
            * self.split.element_bytes
        )
        if expected_payload != self.transport.max_payload_bytes:
            raise OffloadError("offload transport max payload does not match")
        expected_layers = f"{self.split.layer_ids[0]}-{self.split.layer_ids[-1]}"
        if self.backend.layer_spec != expected_layers:
            raise OffloadError("offload backend layer range does not match")

    @classmethod
    def from_json(cls, value: object) -> "OperatorOffloadContract":
        if type(value) is not dict:
            raise OffloadError("operator offload must be an object")
        row: Mapping[str, Any] = value
        if row.get("schema") != OFFLOAD_CONTRACT_SCHEMA:
            raise OffloadError("operator offload schema mismatch")
        return cls(
            route_id=row.get("route_id"),
            host_resource_id=row.get("host_resource_id"),
            split=OperatorSplitPolicy.from_json(row.get("split")),
            transport=PhoneTransportContract.from_json(row.get("transport")),
            backend=PhoneBackendConfig.from_json(row.get("backend")),
        )

    def invocation(self, layer: int, tokens: int) -> OperatorOffloadInvocation:
        split = self.split.invocation(layer, tokens)
        resources = (
            self.host_resource_id,
            *self.transport.required_resource_ids(),
        )
        return OperatorOffloadInvocation(
            split=split,
            transfer=self.transport.transfer(split),
            required_resource_ids=resources,
        )

    def server_environment(self) -> Mapping[str, str]:
        values = {
            "S41_SERVER_FFN_COLUMNS": str(self.split.eligible_columns),
            "S41_SERVER_FFN_F16_IO": "1" if self.split.io_type == "f16" else "0",
            "S41_SERVER_FFN_ACTIVATION": self.split.activation,
            "S41_SERVER_FFN_HOST": self.backend.bridge_bind,
            "S41_SERVER_FFN_LAYER_MASK": self._layer_mask(),
            "S41_SERVER_FFN_N_EMBD": str(self.split.n_embd),
            "S41_SERVER_FFN_POLICY": self.split.table,
            "S41_SERVER_FFN_PORT": str(self.backend.bridge_port),
            "S41_SERVER_FFN_TIMEOUT_MS": str(self.backend.server_timeout_ms),
        }
        return MappingProxyType(values)

    def phone_environment(self) -> Mapping[str, str]:
        values = {
            "S41_FFN_ALTERNATE_COLUMNS": str(
                self.split.alternate_columns[0]
                if self.split.alternate_columns else 0
            ),
            "S41_FFN_COLUMN_QUANTUM": str(self.split.column_quantum),
            "S41_FFN_F16_IO": "1" if self.split.io_type == "f16" else "0",
            "S41_FFN_MAX_TOKENS": str(self.split.max_tokens),
            "S41_FFN_STAGED_DMABUF": (
                "1" if self.transport.allocator.endswith("-split") else "0"
            ),
        }
        return MappingProxyType(values)

    def bridge_command(self) -> tuple[str, ...]:
        return (
            self.backend.artifact("bridge").path,
            self.backend.bridge_bind,
            str(self.backend.bridge_port),
            self.transport.allocator,
        )

    def phone_session_command(self, session_root: str) -> tuple[str, ...]:
        if not _text("phone session_root", session_root).startswith("/"):
            raise OffloadError("phone session_root must be absolute")
        return (
            self.backend.artifact("phone_session").path,
            self.backend.artifact("phone_worker").path,
            self.backend.artifact("phone_model").path,
            self.backend.layer_spec,
            str(self.split.eligible_columns),
            self.backend.compute_backend,
            session_root,
            self.backend.artifact("restore_usb").path,
            str(self.backend.session_timeout_s),
            str(self.backend.max_requests),
        )

    def _layer_mask(self) -> str:
        mask = 0
        for layer in self.split.layer_ids:
            mask |= 1 << layer
        width = max(16, (max(self.split.layer_ids) + 4) // 4)
        return f"0x{mask:0{width}x}"

    def to_json(self) -> dict[str, object]:
        return {
            "backend": self.backend.to_json(),
            "host_resource_id": self.host_resource_id,
            "route_id": self.route_id,
            "schema": OFFLOAD_CONTRACT_SCHEMA,
            "split": self.split.to_json(),
            "transport": self.transport.to_json(),
        }
