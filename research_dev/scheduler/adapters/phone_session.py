"""Canonical lifecycle for a scheduler-selected direct phone FFN session."""

from __future__ import annotations

from pathlib import Path
import socket  # patch anchor: tests patch phone_session.socket.create_connection
import subprocess
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from .._internal.model_manifest import ModelManifest
from .._internal.runtime_plan import (
    PhoneSessionReplacementAuthorization,
    RuntimePhoneShard,
    phone_session_map_sha256,
)
from .bridge import (
    AndroidUsbRestorationReceipt,
    FunctionFsUsbObservation,
    probe_functionfs_usb_device,
    verify_android_usb_restored,
)
from .contracts import PhysicalAdapterError
from .llama_server import (
    LlamaServerPhoneSessionProof,
    PhoneFfnExecutionContract,
    phone_ffn_resident_contract,
)
from .ffn_shards import (
    FfnShardIndex,
    FfnShardIndexError,
    FfnShardRecord,
    remote_hash_entries as ffn_shard_hash_entries,
    verify_remote_hashes as verify_ffn_shard_hashes,
)
from .phone_transport import PhoneTransportContract
from .remote_hash_cache import RemoteFileIdentity, cached_remote_hashes, update_remote_hash_cache
from .ticket import PhysicalTransitionCommand
from .transport_profiles import TransportQualificationIdentity


from .phone_session_contracts.common import (
    _TERMINAL_PATTERN as _TERMINAL_PATTERN,
    _MULTI_TERMINAL_PREFIX as _MULTI_TERMINAL_PREFIX,
    _RESIDENCY_PHASE_PREFIX as _RESIDENCY_PHASE_PREFIX,
    _RESIDENCY_CALL_PREFIX as _RESIDENCY_CALL_PREFIX,
    _android_path as _android_path,
    _artifact_mapping as _artifact_mapping,
)
from .phone_session_contracts.configuration import (
    DirectPhoneFfnSessionConfiguration as DirectPhoneFfnSessionConfiguration,
)
from .phone_session_contracts.events import (
    DirectPhoneFfnSessionProof as DirectPhoneFfnSessionProof,
    DirectPhoneFfnTerminalReceipt as DirectPhoneFfnTerminalReceipt,
    parse_direct_phone_ffn_terminal as parse_direct_phone_ffn_terminal,
    PhoneResidencyPhaseEvent as PhoneResidencyPhaseEvent,
    parse_phone_residency_phase_events as parse_phone_residency_phase_events,
    PhoneResidencyCallEvent as PhoneResidencyCallEvent,
    parse_phone_residency_call_events as parse_phone_residency_call_events,
)
from .phone_session_contracts.receipts import (
    PhoneFfnWeightSource as PhoneFfnWeightSource,
    DirectPhoneFfnLaunchReceipt as DirectPhoneFfnLaunchReceipt,
    DirectPhoneFfnReconfigurationReceipt as DirectPhoneFfnReconfigurationReceipt,
    DirectPhoneFfnCloseReceipt as DirectPhoneFfnCloseReceipt,
    DirectPhoneFfnPreflightReceipt as DirectPhoneFfnPreflightReceipt,
)

from .phone_session_ops.common import (
    _MINIMUM_PERSISTENT_HASH_CACHE_BYTES as _MINIMUM_PERSISTENT_HASH_CACHE_BYTES,
    _ShardResidencyWindow as _ShardResidencyWindow,
)
from .phone_session_ops import completion as _completion
from .phone_session_ops import identity as _identity
from .phone_session_ops import launch as _launch
from .phone_session_ops import preflight as _preflight
from .phone_session_ops import replacement as _replacement
from .phone_session_ops import transport as _transport
from .phone_session_ops import weights as _weights

__all__ = [
    'socket',  # patch anchor for tests
    'AndroidUsbRestorationReceipt',
    'DirectPhoneFfnCloseReceipt',
    'DirectPhoneFfnLaunchReceipt',
    'DirectPhoneFfnPreflightReceipt',
    'DirectPhoneFfnReconfigurationReceipt',
    'DirectPhoneFfnSession',
    'DirectPhoneFfnSessionConfiguration',
    'DirectPhoneFfnSessionProof',
    'DirectPhoneFfnTerminalReceipt',
    'FfnShardIndex',
    'FfnShardIndexError',
    'FfnShardRecord',
    'FunctionFsUsbObservation',
    'LlamaServerPhoneSessionProof',
    'ModelManifest',
    'PhoneFfnExecutionContract',
    'PhoneFfnWeightSource',
    'PhoneResidencyCallEvent',
    'PhoneResidencyPhaseEvent',
    'PhoneSessionReplacementAuthorization',
    'PhoneTransportContract',
    'PhysicalAdapterError',
    'PhysicalTransitionCommand',
    'RemoteFileIdentity',
    'RuntimePhoneShard',
    'TransportQualificationIdentity',
    '_MINIMUM_PERSISTENT_HASH_CACHE_BYTES',
    '_MULTI_TERMINAL_PREFIX',
    '_RESIDENCY_CALL_PREFIX',
    '_RESIDENCY_PHASE_PREFIX',
    '_ShardResidencyWindow',
    '_TERMINAL_PATTERN',
    '_android_path',
    '_artifact_mapping',
    '_completion',
    '_identity',
    '_launch',
    '_preflight',
    '_replacement',
    '_transport',
    '_weights',
    'cached_remote_hashes',
    'ffn_shard_hash_entries',
    'parse_direct_phone_ffn_terminal',
    'parse_phone_residency_call_events',
    'parse_phone_residency_phase_events',
    'phone_ffn_resident_contract',
    'phone_session_map_sha256',
    'probe_functionfs_usb_device',
    'update_remote_hash_cache',
    'verify_android_usb_restored',
    'verify_ffn_shard_hashes',
]


class DirectPhoneFfnSession:
    """Start and collect one direct DMA-BUF worker selected by a ticket."""

    def __init__(
        self,
        configuration: DirectPhoneFfnSessionConfiguration,
    ) -> None:
        if not isinstance(
            configuration, DirectPhoneFfnSessionConfiguration
        ):
            raise PhysicalAdapterError("direct phone session is invalid")
        self.configuration = configuration
        self._launch: DirectPhoneFfnLaunchReceipt | None = None
        self._remote_root: str | None = None
        self._bound_ticket_ids: list[str] = []
        self._proof_shards: list[RuntimePhoneShard] = []
        self._executed_proof_shards: list[RuntimePhoneShard] = []
        self._residency_generation = 0
        self._shard_residency_windows: list[_ShardResidencyWindow] = []
        self._ticket_bind_generations: dict[str, int] = {}
        self._historical_execution_proofs: list[dict[str, object]] = []
        self._verified_remote_hash_by_path: dict[str, str] = {}
        self._execution_by_artifact: dict[
            str, PhoneFfnExecutionContract
        ] = {}
        self._transport_by_artifact: dict[
            str, PhoneTransportContract
        ] = {}
        self._load_count_by_session: dict[str, int] = {}
        self._column_quantum_by_session: dict[str, int] = {}
        self._max_tokens_by_session: dict[str, int] = {}
        self._multi_session_port_by_id: dict[str, int] = {}

    def _remote_file_identities(
        self,
        paths: tuple[str, ...],
        *,
        root: bool,
        timeout_s: int,
    ) -> Mapping[str, RemoteFileIdentity]:
        return _identity._remote_file_identities(self, paths, root=root, timeout_s=timeout_s)

    @property
    def active(self) -> bool:
        return self._launch is not None and self._remote_root is not None

    @property
    def phone_shards(self) -> tuple[RuntimePhoneShard, ...]:
        launch = self._launch
        return () if launch is None else launch.phone_shards

    @property
    def execution_by_artifact(
        self,
    ) -> Mapping[str, PhoneFfnExecutionContract]:
        return MappingProxyType(dict(self._execution_by_artifact))

    @property
    def residency_generation(self) -> int:
        return self._residency_generation

    @property
    def load_count_by_session(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._load_count_by_session))

    @property
    def column_quantum_by_session(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._column_quantum_by_session))

    @property
    def max_tokens_by_session(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._max_tokens_by_session))

    @property
    def weight_sources(self) -> tuple[PhoneFfnWeightSource, ...]:
        launch = self._launch
        return () if launch is None else launch.weight_sources

    def _current_close_identities(
        self,
    ) -> tuple[
        tuple[
            PhoneFfnExecutionContract,
            PhoneTransportContract,
            str,
        ],
        ...,
    ]:
        return _identity._current_close_identities(self)

    def _current_close_identity(
        self,
    ) -> tuple[
        PhoneFfnExecutionContract,
        PhoneTransportContract,
        str,
    ]:
        return _identity._current_close_identity(self)

    def _close_current_usb(self) -> None:
        return _identity._close_current_usb(self)

    def supports(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        transport: PhoneTransportContract,
    ) -> bool:
        return _identity.supports(self, command, manifest, transport)

    def bind(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        transport: PhoneTransportContract,
    ) -> None:
        return _identity.bind(self, command, manifest, transport)

    def bind_ticket_generation(self, ticket_id: str) -> int:
        """Pin a ticket's proofs to the residency generation it started in."""
        return _identity.bind_ticket_generation(self, ticket_id)

    @staticmethod
    def _proof_key(
        row: RuntimePhoneShard | LlamaServerPhoneSessionProof,
    ) -> tuple[str, str | None, str, str, int]:
        return _identity._proof_key(row)

    def _replace_resident_shard(
        self,
        previous_shard: RuntimePhoneShard | None,
        target_shard: RuntimePhoneShard,
    ) -> None:
        """Advance physical proof windows for one added or replaced shard."""
        return _replacement._replace_resident_shard(self, previous_shard, target_shard)

    def record_execution_proof(
        self,
        ticket_id: str,
        artifact_sha256: str,
        proofs: tuple[LlamaServerPhoneSessionProof, ...],
    ) -> None:
        return _identity.record_execution_proof(self, ticket_id, artifact_sha256, proofs)

    @staticmethod
    def _changed_phone_sessions(
        current: tuple[RuntimePhoneShard, ...],
        target: tuple[RuntimePhoneShard, ...],
    ) -> tuple[str, ...]:
        return _replacement._changed_phone_sessions(current, target)

    def _validate_partial_reconfiguration_authority(
        self,
        command: PhysicalTransitionCommand,
    ) -> None:
        return _replacement._validate_partial_reconfiguration_authority(self, command)

    def supports_partial_reconfiguration(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        transport: PhoneTransportContract,
    ) -> bool:
        return _replacement.supports_partial_reconfiguration(self, command, manifest, transport)

    def reconfigure(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        transport: PhoneTransportContract,
    ) -> DirectPhoneFfnReconfigurationReceipt:
        return _replacement.reconfigure(self, command, manifest, transport)

    def rollback_reconfiguration(
        self,
        receipt: DirectPhoneFfnReconfigurationReceipt,
    ) -> None:
        return _replacement.rollback_reconfiguration(self, receipt)

    def _adb(self, remote_command: str, timeout_s: int = 30) -> str:
        return _transport._adb(self, remote_command, timeout_s)

    def _adb_root(self, remote_command: str, timeout_s: int = 30) -> str:
        return _transport._adb_root(self, remote_command, timeout_s)

    def _remote_hashes(
        self,
        paths: Mapping[str, str],
        *,
        root: bool = False,
        timeout_s: int = 30,
    ) -> Mapping[str, str]:
        return _transport._remote_hashes(self, paths, root=root, timeout_s=timeout_s)

    @staticmethod
    def _sha256(path: Path) -> str:
        return _transport._sha256(path)

    def _close_direct_usb(
        self,
        execution: PhoneFfnExecutionContract,
        transport: PhoneTransportContract,
        artifact_sha256: str = "sha256:" + "0" * 64,
    ) -> None:
        return _transport._close_direct_usb(self, execution, transport, artifact_sha256)

    def _remote_launch_failure(self, remote_root: str) -> str | None:
        return _transport._remote_launch_failure(self, remote_root)

    def _phone_kernel_release(self) -> str:
        return _transport._phone_kernel_release(self)

    def _validate_static_transport_identity(
        self,
        remote_hashes: Mapping[str, str],
    ) -> None:
        return _transport._validate_static_transport_identity(self, remote_hashes)

    def _validate_live_transport_identity(
        self,
        transport: PhoneTransportContract,
        usb: FunctionFsUsbObservation,
        phone_kernel_release: str,
        phone_usb_controller: str,
    ) -> None:
        return _transport._validate_live_transport_identity(
            self,
            transport,
            usb,
            phone_kernel_release,
            phone_usb_controller,
        )

    @staticmethod
    def _ncm_interfaces(sysfs_device: str) -> tuple[str, ...]:
        return _transport._ncm_interfaces(sysfs_device)

    def _diagnostic_available(self) -> bool:
        return _transport._diagnostic_available(self)

    def _read_diagnostic_file(self, name: str) -> str:
        return _transport._read_diagnostic_file(self, name)

    def residency_phase_events(
        self,
    ) -> tuple[PhoneResidencyPhaseEvent, ...]:
        return _transport.residency_phase_events(self)

    def residency_call_events(
        self,
    ) -> tuple[PhoneResidencyCallEvent, ...]:
        return _transport.residency_call_events(self)

    def _connect_diagnostic_ncm(
        self, usb: FunctionFsUsbObservation
    ) -> str | None:
        return _transport._connect_diagnostic_ncm(self, usb)

    @staticmethod
    def _worker_environment(
        execution: PhoneFfnExecutionContract,
        transport: PhoneTransportContract,
        column_quantum: int,
    ) -> tuple[str, ...]:
        return _weights._worker_environment(execution, transport, column_quantum)

    @staticmethod
    def _layer_spec(mask: int) -> str:
        return _weights._layer_spec(mask)

    def _resolve_phone_weight_source(
        self,
        shard: RuntimePhoneShard,
        phase_events: tuple[PhoneResidencyPhaseEvent, ...] = (),
    ) -> PhoneFfnWeightSource:
        return _weights._resolve_phone_weight_source(self, shard, phase_events)

    def _phone_weight_sources(
        self,
        shards: tuple[RuntimePhoneShard, ...],
        phase_events: tuple[PhoneResidencyPhaseEvent, ...] = (),
    ) -> tuple[PhoneFfnWeightSource, ...]:
        return _weights._phone_weight_sources(self, shards, phase_events)

    def _multi_session_manifest(
        self,
        shards: tuple[RuntimePhoneShard, ...],
    ) -> tuple[str, str]:
        return _weights._multi_session_manifest(self, shards)

    @staticmethod
    def _validate_shard_loads(
        lines: tuple[str, ...] | list[str],
        shards: tuple[RuntimePhoneShard, ...],
    ) -> None:
        return _weights._validate_shard_loads(lines, shards)

    @staticmethod
    def _validate_shard_terminal(
        terminal: DirectPhoneFfnTerminalReceipt,
        shards: tuple[RuntimePhoneShard, ...],
        *,
        require_execution: bool = True,
        executed_shards: tuple[RuntimePhoneShard, ...] | None = None,
        historical_shards: tuple[RuntimePhoneShard, ...] = (),
    ) -> None:
        return _weights._validate_shard_terminal(
            terminal,
            shards,
            require_execution=require_execution,
            executed_shards=executed_shards,
            historical_shards=historical_shards,
        )

    def preflight(self) -> DirectPhoneFfnPreflightReceipt:
        return _preflight.preflight(self)

    def _validate_worker_loadability(
        self, workers: Iterable[str]
    ) -> None:
        return _preflight._validate_worker_loadability(self, workers)

    def _start_contract(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        transport: PhoneTransportContract,
        control_check: Callable[[], None] | None,
    ) -> tuple[
        Callable[[], None],
        str,
        PhoneFfnExecutionContract,
        tuple[RuntimePhoneShard, ...],
        bool,
        str | None,
        str | None,
        str,
        str,
        str,
    ]:
        return _preflight._start_contract(self, command, manifest, transport, control_check)

    def _start_remote_hashes(
        self,
        manifest: ModelManifest,
        shards: tuple[RuntimePhoneShard, ...],
        worker: str,
        model: str,
        multi_session: bool,
        control_check: Callable[[], None],
    ) -> tuple[Mapping[str, str], str]:
        return _preflight._start_remote_hashes(
            self,
            manifest,
            shards,
            worker,
            model,
            multi_session,
            control_check,
        )

    def _start_remote_command(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        execution: PhoneFfnExecutionContract,
        transport: PhoneTransportContract,
        shards: tuple[RuntimePhoneShard, ...],
        multi_session: bool,
        shard_manifest: str | None,
        worker: str,
        model: str,
        backend: str,
    ) -> tuple[str, str, str, str]:
        return _launch._start_remote_command(
            self,
            command,
            manifest,
            execution,
            transport,
            shards,
            multi_session,
            shard_manifest,
            worker,
            model,
            backend,
        )

    def _append_start_environment(self, words: list[str]) -> None:
        return _launch._append_start_environment(self, words)

    def _spawn_phone_session(
        self,
        remote_command: str,
        control_check: Callable[[], None],
    ) -> subprocess.Popen[str]:
        return _launch._spawn_phone_session(self, remote_command, control_check)

    def _remote_start_failure_detail(
        self, remote_root: str, launch_log: str
    ) -> str | None:
        return _launch._remote_start_failure_detail(self, remote_root, launch_log)

    def _wait_for_phone_session(
        self,
        process: subprocess.Popen[str],
        transport: PhoneTransportContract,
        shards: tuple[RuntimePhoneShard, ...],
        phone_kernel_release: str,
        phone_usb_controller: str,
        remote_root: str,
        launch_log: str,
        control_check: Callable[[], None],
    ) -> tuple[FunctionFsUsbObservation | None, str]:
        return _launch._wait_for_phone_session(
            self,
            process,
            transport,
            shards,
            phone_kernel_release,
            phone_usb_controller,
            remote_root,
            launch_log,
            control_check,
        )

    @staticmethod
    def _stop_phone_session_launcher(process: subprocess.Popen[str]) -> None:
        return _launch._stop_phone_session_launcher(process)

    def _fail_phone_session_start(
        self,
        last_error: str,
        remote_root: str,
        launch_log: str,
        execution: PhoneFfnExecutionContract,
        transport: PhoneTransportContract,
        artifact_sha256: str,
    ) -> None:
        return _launch._fail_phone_session_start(
            self,
            last_error,
            remote_root,
            launch_log,
            execution,
            transport,
            artifact_sha256,
        )

    def _connect_start_diagnostic(
        self,
        usb: FunctionFsUsbObservation,
        execution: PhoneFfnExecutionContract,
        transport: PhoneTransportContract,
        artifact_sha256: str,
        control_check: Callable[[], None],
    ) -> str | None:
        return _launch._connect_start_diagnostic(
            self,
            usb,
            execution,
            transport,
            artifact_sha256,
            control_check,
        )

    def _publish_phone_session_launch(
        self,
        command: PhysicalTransitionCommand,
        execution: PhoneFfnExecutionContract,
        transport: PhoneTransportContract,
        usb: FunctionFsUsbObservation,
        remote_hashes: Mapping[str, str],
        phone_kernel_release: str,
        diagnostic_interface: str | None,
        shards: tuple[RuntimePhoneShard, ...],
        multi_session: bool,
        shard_manifest_sha256: str | None,
        token: str,
        remote_root: str,
    ) -> DirectPhoneFfnLaunchReceipt:
        return _launch._publish_phone_session_launch(
            self,
            command,
            execution,
            transport,
            usb,
            remote_hashes,
            phone_kernel_release,
            diagnostic_interface,
            shards,
            multi_session,
            shard_manifest_sha256,
            token,
            remote_root,
        )

    def start(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        transport: PhoneTransportContract,
        control_check: Callable[[], None] | None = None,
    ) -> DirectPhoneFfnLaunchReceipt:
        return _launch.start(self, command, manifest, transport, control_check)

    def finish(
        self, *, require_execution: bool = True
    ) -> DirectPhoneFfnCloseReceipt:
        return _completion.finish(self, require_execution=require_execution)

    def abort(self) -> AndroidUsbRestorationReceipt:
        return _completion.abort(self)
