"""Provision one qualified static operator-split endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import time
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import PhysicalAdapterError
from .http_backend import LlamaCppCompletionPayload, LlamaCppHttpClient
from .split_contract import (
    build_static_split_prewarm,
    validate_static_split_capability,
)


@dataclass(frozen=True)
class StaticSplitLaunchContract:
    environment: Mapping[str, str]
    policy_text: str

    def __post_init__(self) -> None:
        environment = dict(self.environment)
        if (
            not environment
            or any(
                type(name) is not str
                or not name
                or not name.isascii()
                or type(value) is not str
                or not value
                or not value.isascii()
                for name, value in environment.items()
            )
            or type(self.policy_text) is not str
            or not self.policy_text
            or not self.policy_text.isascii()
        ):
            raise PhysicalAdapterError("static split launch contract is invalid")
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(dict(sorted(environment.items()))),
        )


@dataclass(frozen=True)
class StaticSplitPrewarmReceipt:
    executor_id: str
    endpoint: str
    input_tokens: int
    phone_columns: int
    started_ns: int
    finished_ns: int
    stream_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.executor_id) is not str
            or not self.executor_id
            or not self.executor_id.isascii()
            or type(self.endpoint) is not str
            or not self.endpoint
            or not self.endpoint.isascii()
            or type(self.input_tokens) is not int
            or self.input_tokens <= 0
            or type(self.phone_columns) is not int
            or self.phone_columns <= 0
            or type(self.started_ns) is not int
            or self.started_ns < 0
            or type(self.finished_ns) is not int
            or self.finished_ns < self.started_ns
            or type(self.stream_sha256) is not str
            or len(self.stream_sha256) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.stream_sha256
            )
        ):
            raise PhysicalAdapterError("static split prewarm receipt is invalid")

    @property
    def duration_ms(self) -> float:
        return (self.finished_ns - self.started_ns) / 1e6

    def to_json(self) -> dict[str, object]:
        return {
            "duration_ms": self.duration_ms,
            "endpoint": self.endpoint,
            "executor_id": self.executor_id,
            "finished_ns": self.finished_ns,
            "input_tokens": self.input_tokens,
            "phone_columns": self.phone_columns,
            "started_ns": self.started_ns,
            "stream_sha256": self.stream_sha256,
        }


def static_split_launch_contract(
    manifest: Mapping[str, object],
    policy: Mapping[str, object],
    *,
    phone_host: str,
    phone_port: int,
    timeout_ms: int = 600_000,
) -> StaticSplitLaunchContract:
    """Translate validated static-split evidence into server environment data."""
    split = manifest.get("split_contract")
    geometry = manifest.get("geometry")
    policy_text = policy.get("policy_text")
    if (
        type(split) is not dict
        or type(split.get("max_columns")) is not int
        or split["max_columns"] <= 0
        or type(split.get("layer_mask")) is not int
        or split["layer_mask"] <= 0
        or type(geometry) is not dict
        or type(geometry.get("n_embd")) is not int
        or geometry["n_embd"] <= 0
        or type(policy_text) is not str
        or not policy_text
        or not policy_text.isascii()
        or type(phone_host) is not str
        or not phone_host
        or not phone_host.isascii()
        or type(phone_port) is not int
        or not 0 < phone_port < 65_536
        or type(timeout_ms) is not int
        or timeout_ms <= 0
    ):
        raise PhysicalAdapterError("static split launch evidence is invalid")
    return StaticSplitLaunchContract(
        environment={
            "S41_SERVER_FFN_ACTIVATION": "swiglu",
            "S41_SERVER_FFN_COLUMNS": str(split["max_columns"]),
            "S41_SERVER_FFN_F16_IO": "1",
            "S41_SERVER_FFN_HOST": phone_host,
            "S41_SERVER_FFN_LAYER_MASK": str(split["layer_mask"]),
            "S41_SERVER_FFN_N_EMBD": str(geometry["n_embd"]),
            "S41_SERVER_FFN_POLICY": policy_text,
            "S41_SERVER_FFN_PORT": str(phone_port),
            "S41_SERVER_FFN_TIMEOUT_MS": str(timeout_ms),
        },
        policy_text=policy_text,
    )


def validate_static_split_ready_log(
    lines: Sequence[str], contract: StaticSplitLaunchContract
) -> None:
    if not isinstance(contract, StaticSplitLaunchContract) or any(
        type(line) is not str for line in lines
    ):
        raise PhysicalAdapterError("static split readiness evidence is invalid")
    if not any(
        "S41SERVERFFN ready " in line
        and "policy=" + contract.policy_text in line
        for line in lines
    ):
        raise PhysicalAdapterError(
            "static split readiness differs from the launch contract"
        )


class CanonicalStaticSplitPrewarmer:
    """Validate and prewarm one catalog-declared static split endpoint."""

    def __init__(self, client: LlamaCppHttpClient) -> None:
        if not isinstance(client, LlamaCppHttpClient):
            raise PhysicalAdapterError("static split HTTP client is invalid")
        self._client = client

    def prewarm(
        self,
        capability: object,
        *,
        artifact_sha256: str,
        manifest: Mapping[str, object],
        policy: Mapping[str, object],
        rows: Sequence[Mapping[str, object]],
        event_id: str,
        request_index: int,
        expected_model_alias: str,
        stream_path: Path,
        timeout_s: float,
    ) -> StaticSplitPrewarmReceipt:
        protocol = getattr(capability, "operator_plan_protocol", None)
        if type(protocol) is not str or ":" not in protocol:
            raise PhysicalAdapterError("static split protocol is invalid")
        namespace, _, suffix = protocol.rpartition(":")
        if (
            not namespace
            or len(suffix) != 64
            or any(value not in "0123456789abcdef" for value in suffix)
        ):
            raise PhysicalAdapterError("static split protocol is invalid")
        endpoint = getattr(capability, "endpoint", None)
        validate_static_split_capability(
            capability,
            endpoint=endpoint,
            artifact_sha256=artifact_sha256,
            manifest=manifest,
            policy=policy,
            protocol_namespace=namespace,
        )
        row, shape = build_static_split_prewarm(
            rows,
            policy,
            event_id=event_id,
            request_index=request_index,
        )
        started_ns = time.monotonic_ns()
        result = self._client.complete(
            endpoint,
            LlamaCppCompletionPayload(
                request_id=event_id,
                expected_model_alias=expected_model_alias,
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                prompt_tokens=tuple(row["prompt_tokens"]),
                seed=request_index,
                stream_path=stream_path,
                on_first_token=lambda _: None,
                timeout_s=timeout_s,
            ),
            lambda: None,
        )
        finished_ns = time.monotonic_ns()
        return StaticSplitPrewarmReceipt(
            executor_id=getattr(capability, "executor_id", None),
            endpoint=endpoint,
            input_tokens=shape["input_tokens"],
            phone_columns=shape["phone_columns"],
            started_ns=started_ns,
            finished_ns=finished_ns,
            stream_sha256=result["stream_sha256"],
        )
