"""Track physical executions without introducing route policy."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from types import MappingProxyType
from typing import Mapping

from .._internal.runtime_capabilities import RuntimeCapabilityCatalog
from .contracts import PhysicalAdapterError
from .ticket import PhysicalExecutionCommand


@dataclass(frozen=True)
class RuntimeActivitySnapshot:
    active_by_device_id: Mapping[str, int]
    active_by_device_kind: Mapping[str, int]
    active_by_executor_id: Mapping[str, int]
    active_input_tokens_by_model: Mapping[str, int]
    active_output_tokens_by_model: Mapping[str, int]
    active_requests_by_model: Mapping[str, int]

    def __post_init__(self) -> None:
        for name in (
            "active_by_device_id",
            "active_by_device_kind",
            "active_by_executor_id",
            "active_input_tokens_by_model",
            "active_output_tokens_by_model",
            "active_requests_by_model",
        ):
            values = dict(getattr(self, name))
            if any(
                type(key) is not str
                or not key
                or not key.isascii()
                or type(value) is not int
                or value < 0
                for key, value in values.items()
            ):
                raise PhysicalAdapterError("runtime activity is invalid")
            object.__setattr__(
                self, name, MappingProxyType(dict(sorted(values.items())))
            )


class RuntimeActivityTracker:
    """Maintain raw active-execution counts for runtime observations."""

    def __init__(self, catalog: RuntimeCapabilityCatalog) -> None:
        if not isinstance(catalog, RuntimeCapabilityCatalog):
            raise PhysicalAdapterError("runtime activity catalog is invalid")
        self._kind_by_device = {
            device_id: device.kind
            for device_id, device in catalog.placement_profile.devices.items()
        }
        self._active_commands: dict[
            str, tuple[PhysicalExecutionCommand, int, int]
        ] = {}
        self._lock = threading.Lock()

    def start(
        self,
        command: PhysicalExecutionCommand,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        if (
            not isinstance(command, PhysicalExecutionCommand)
            or type(input_tokens) is not int
            or input_tokens < 0
            or type(output_tokens) is not int
            or output_tokens < 0
        ):
            raise PhysicalAdapterError("runtime activity command is invalid")
        with self._lock:
            if command.ticket_id in self._active_commands:
                raise PhysicalAdapterError(
                    "runtime activity command is already active"
                )
            self._active_commands[command.ticket_id] = (
                command, input_tokens, output_tokens
            )

    def finish(self, command: PhysicalExecutionCommand) -> None:
        if not isinstance(command, PhysicalExecutionCommand):
            raise PhysicalAdapterError("runtime activity command is invalid")
        with self._lock:
            active = self._active_commands.get(command.ticket_id)
            if active is None or active[0] != command:
                raise PhysicalAdapterError(
                    "runtime activity completion differs from its start"
                )
            del self._active_commands[command.ticket_id]

    def snapshot(self) -> RuntimeActivitySnapshot:
        with self._lock:
            active = tuple(self._active_commands.values())
        by_device: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        by_executor: dict[str, int] = {}
        input_by_model: dict[str, int] = {}
        output_by_model: dict[str, int] = {}
        requests_by_model: dict[str, int] = {}
        for command, input_tokens, output_tokens in active:
            by_executor[command.executor_id] = (
                by_executor.get(command.executor_id, 0) + 1
            )
            for participant in command.participants:
                by_device[participant.device_id] = (
                    by_device.get(participant.device_id, 0) + 1
                )
                kind = self._kind_by_device.get(participant.device_id)
                if kind is None:
                    raise PhysicalAdapterError(
                        "runtime activity participant device is unknown"
                    )
                by_kind[kind] = by_kind.get(kind, 0) + 1
            input_by_model[command.model_id] = (
                input_by_model.get(command.model_id, 0) + input_tokens
            )
            output_by_model[command.model_id] = (
                output_by_model.get(command.model_id, 0) + output_tokens
            )
            requests_by_model[command.model_id] = (
                requests_by_model.get(command.model_id, 0) + 1
            )
        return RuntimeActivitySnapshot(
            active_by_device_id=by_device,
            active_by_device_kind=by_kind,
            active_by_executor_id=by_executor,
            active_input_tokens_by_model=input_by_model,
            active_output_tokens_by_model=output_by_model,
            active_requests_by_model=requests_by_model,
        )
