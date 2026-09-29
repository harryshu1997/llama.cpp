"""Validate llama-server launch data carried by a scheduler ticket."""

from __future__ import annotations

from dataclasses import dataclass, replace
import http.client
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit

from .._internal.adaptive_decode_contracts import (
    AdaptiveDecodeGroupedObservation,
)
from .._internal.model_manifest import ModelManifest, _sha256_file
from .._internal.runtime_capabilities import RuntimeExecutorCapability
from .._internal.runtime_plan import RuntimeHelperExecutionEnvelope
from .._internal.runtime_resources import RuntimeRemoteResidentOmissionProof
from .._internal.types import canonical_sha256
from .contracts import (
    LlamaServerExitedError,
    PhysicalAdapterError,
    dormant_phone_ffn_parameters,
)
from .phone_transport import phone_transport_contract
from .ticket import (
    PhysicalExecutionCommand,
    PhysicalTransitionCommand,
    static_decode_policy,
    validate_physical_execution_command,
)

from .llama_server_ops.proofs import ManagedServerProofMixin
from .llama_server_contracts import (
    llama_server_runtime_timing as llama_server_runtime_timing,
    _PhoneFfnCommandView as _PhoneFfnCommandView,
    _phone_ffn_command as _phone_ffn_command,
    _adaptive_phone_ffn as _adaptive_phone_ffn,
    LlamaServerLaunchContract as LlamaServerLaunchContract,
    _launch_contract_supports_execution as _launch_contract_supports_execution,
    PhoneFfnExecutionContract as PhoneFfnExecutionContract,
    LlamaServerFfnCallContext as LlamaServerFfnCallContext,
    LlamaServerFfnCall as LlamaServerFfnCall,
    LlamaServerPhoneSessionProof as LlamaServerPhoneSessionProof,
    LlamaServerExecutionMarker as LlamaServerExecutionMarker,
    LlamaServerExecutionProof as LlamaServerExecutionProof,
    _FFN_CALL_PATTERN as _FFN_CALL_PATTERN,
    _SCOPED_FFN_CALL_PATTERN as _SCOPED_FFN_CALL_PATTERN,
    _LLAMA_LOGGED_FFN_CALL_PREFIX as _LLAMA_LOGGED_FFN_CALL_PREFIX,
    _TAIL_RELEASE_REASONS as _TAIL_RELEASE_REASONS,
    parse_llama_server_ffn_call as parse_llama_server_ffn_call,
    _integer as _integer,
    _text as _text,
    _layer_index as _layer_index,
    _operator_rows as _operator_rows,
    _gpu_layers_from_plan as _gpu_layers_from_plan,
    _phone_ffn_from_plan as _phone_ffn_from_plan,
    phone_ffn_execution_contract as phone_ffn_execution_contract,
    phone_ffn_resident_contract as phone_ffn_resident_contract,
    primary_phone_ffn_contract as primary_phone_ffn_contract,
    _remote_resident_launch_environment as _remote_resident_launch_environment,
    llama_server_launch_contract as llama_server_launch_contract,
    llama_server_capability_contract as llama_server_capability_contract,
)

__all__ = [
    'AdaptiveDecodeGroupedObservation',
    'LlamaServerExecutionMarker',
    'LlamaServerExecutionProof',
    'LlamaServerFfnCall',
    'LlamaServerFfnCallContext',
    'LlamaServerLaunchContract',
    'LlamaServerPhoneSessionProof',
    'LlamaServerProcessConfiguration',
    'LlamaServerProcessLauncher',
    'ManagedLlamaServer',
    'ManagedServerProofMixin',
    'ModelManifest',
    'PhoneFfnExecutionContract',
    'PhysicalAdapterError',
    'PhysicalExecutionCommand',
    'PhysicalTransitionCommand',
    'RuntimeExecutorCapability',
    'RuntimeHelperExecutionEnvelope',
    'RuntimeRemoteResidentOmissionProof',
    '_FFN_CALL_PATTERN',
    '_LLAMA_LOGGED_FFN_CALL_PREFIX',
    '_PhoneFfnCommandView',
    '_SCOPED_FFN_CALL_PATTERN',
    '_TAIL_RELEASE_REASONS',
    '_adaptive_phone_ffn',
    '_gpu_layers_from_plan',
    '_integer',
    '_launch_contract_supports_execution',
    '_layer_index',
    '_operator_rows',
    '_phone_ffn_command',
    '_phone_ffn_from_plan',
    '_remote_resident_launch_environment',
    '_text',
    'canonical_sha256',
    'dormant_phone_ffn_parameters',
    'llama_server_capability_contract',
    'llama_server_launch_contract',
    'llama_server_runtime_timing',
    'parse_llama_server_ffn_call',
    'phone_ffn_execution_contract',
    'phone_ffn_resident_contract',
    'phone_transport_contract',
    'primary_phone_ffn_contract',
    'static_decode_policy',
    'validate_physical_execution_command',
]














_FFN_PROOF_DRAIN_TIMEOUT_S = 5.0
_STARTUP_MARKER_TIMEOUT_S = 30.0













@dataclass(frozen=True)
class LlamaServerProcessConfiguration:
    """Rig paths used to execute a scheduler-selected launch contract."""

    server_path: Path
    model_paths_by_artifact: Mapping[str, Path]
    library_paths_by_device: Mapping[str, tuple[Path, ...]]
    executable_device_names: Mapping[str, str]
    output_directory: Path
    common_library_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.server_path, Path)
            or not self.server_path.is_file()
            or not os.access(self.server_path, os.X_OK)
            or not isinstance(self.output_directory, Path)
            or not self.output_directory.is_dir()
        ):
            raise PhysicalAdapterError(
                "llama-server process configuration is invalid"
            )
        models = dict(self.model_paths_by_artifact)
        if not models or any(
            type(artifact) is not str
            or not artifact.startswith("sha256:")
            or len(artifact) != 71
            or not isinstance(path, Path)
            or not path.is_file()
            for artifact, path in models.items()
        ):
            raise PhysicalAdapterError(
                "llama-server model path configuration is invalid"
            )
        libraries = {
            device_id: tuple(paths)
            for device_id, paths in self.library_paths_by_device.items()
        }
        common_libraries = tuple(self.common_library_paths)
        if any(
            type(device_id) is not str
            or not device_id
            or not device_id.isascii()
            or any(not isinstance(path, Path) or not path.is_dir()
                   for path in paths)
            for device_id, paths in libraries.items()
        ) or any(
            not isinstance(path, Path) or not path.is_dir()
            for path in common_libraries
        ):
            raise PhysicalAdapterError(
                "llama-server library path configuration is invalid"
            )
        names = dict(self.executable_device_names)
        if any(
            type(device_id) is not str
            or not device_id
            or not device_id.isascii()
            or type(name) is not str
            or not name
            or not name.isascii()
            for device_id, name in names.items()
        ):
            raise PhysicalAdapterError(
                "llama-server device name configuration is invalid"
            )
        object.__setattr__(
            self,
            "model_paths_by_artifact",
            MappingProxyType(dict(sorted(models.items()))),
        )
        object.__setattr__(
            self,
            "library_paths_by_device",
            MappingProxyType(dict(sorted(libraries.items()))),
        )
        object.__setattr__(
            self,
            "common_library_paths",
            tuple(dict.fromkeys(common_libraries)),
        )
        object.__setattr__(
            self,
            "executable_device_names",
            MappingProxyType(dict(sorted(names.items()))),
        )


class ManagedLlamaServer(ManagedServerProofMixin):
    """One exact llama-server process published by a transition ticket."""

    def __init__(
        self,
        command: tuple[str, ...],
        environment: Mapping[str, str],
        output_directory: Path,
        label: str,
        launch_contract: LlamaServerLaunchContract,
    ) -> None:
        if (
            not command
            or any(type(value) is not str or not value for value in command)
            or type(label) is not str
            or not label
            or not label.isascii()
        ):
            raise PhysicalAdapterError(
                "managed llama-server command is invalid"
            )
        self.command = command
        self.environment = dict(environment)
        self.output_directory = output_directory
        self.label = label
        if not isinstance(launch_contract, LlamaServerLaunchContract):
            raise PhysicalAdapterError(
                "managed llama-server launch contract is invalid"
            )
        self.launch_contract = launch_contract
        self.process: subprocess.Popen[str] | None = None
        self.stderr_lines: list[str] = []
        self.stderr_observed_epoch_us: list[int] = []
        self._stderr_lock = threading.Condition()
        self._stderr_thread: threading.Thread | None = None
        self._stderr_file = None
        self._stdout_file = None

    @property
    def pid(self) -> int:
        if self.process is None:
            raise PhysicalAdapterError("managed llama-server is not started")
        return self.process.pid

    def exit_code(self) -> int | None:
        """The process return code once it has exited, else None.

        A server that was never started has no exit code either; callers
        that need liveness check ``process`` first.
        """
        process = self.process
        return None if process is None else process.poll()

    def failure_evidence(
        self,
        stderr_index: int,
        *,
        timeout_s: float,
        decisive: Callable[[Sequence[str]], bool] | None = None,
    ) -> tuple[tuple[str, ...], int | None]:
        """Stderr lines since ``stderr_index`` after a failed request.

        Waits at most ``timeout_s`` for the process to exit and its stderr
        reader to drain (shutdown lines such as ``S41SERVERFFNERROR`` are
        printed while the server exits), or until ``decisive(lines)`` holds.
        Returns the lines and the exit code (None while still running).
        """
        if type(stderr_index) is not int or stderr_index < 0:
            raise PhysicalAdapterError(
                "llama-server failure evidence index is invalid"
            )
        if timeout_s < 0:
            raise PhysicalAdapterError(
                "llama-server failure evidence timeout is invalid"
            )
        if decisive is not None and not callable(decisive):
            raise PhysicalAdapterError(
                "llama-server failure evidence predicate is invalid"
            )
        deadline = time.monotonic() + timeout_s
        while True:
            returncode = self.exit_code()
            reader = self._stderr_thread
            drained = (
                returncode is not None
                and (reader is None or not reader.is_alive())
            )
            with self._stderr_lock:
                lines = tuple(self.stderr_lines[stderr_index:])
                if drained or (
                    decisive is not None and decisive(lines)
                ):
                    return lines, returncode
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return lines, returncode
                self._stderr_lock.wait(min(remaining, 0.05))

    def remote_resident_proof(self) -> RuntimeRemoteResidentOmissionProof | None:
        """The server's omission record, or None when it omitted nothing."""
        proofs = [
            proof
            for line in tuple(self.stderr_lines)
            for proof in (RuntimeRemoteResidentOmissionProof.parse_line(line),)
            if proof is not None
        ]
        if len(proofs) > 1:
            raise PhysicalAdapterError(
                "llama-server emitted several remote-resident proofs"
            )
        return proofs[0] if proofs else None

    def start(self) -> None:
        if self.process is not None:
            raise PhysicalAdapterError("managed llama-server already started")
        self._stderr_file = (
            self.output_directory / (self.label + ".stderr")
        ).open("x", encoding="utf-8")
        self._stdout_file = (
            self.output_directory / (self.label + ".stdout")
        ).open("x", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=self._stdout_file,
            stderr=subprocess.PIPE,
            env=self.environment,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            bufsize=1,
            start_new_session=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name=self.label + "-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        assert self._stderr_file is not None
        for line in self.process.stderr:
            observed_epoch_us = time.time_ns() // 1000
            self._stderr_file.write(line)
            self._stderr_file.flush()
            with self._stderr_lock:
                self.stderr_lines.append(line.rstrip("\n"))
                self.stderr_observed_epoch_us.append(observed_epoch_us)
                self._stderr_lock.notify_all()

    def slot_allocation_events(self) -> tuple[dict[str, object], ...]:
        with self._stderr_lock:
            rows = tuple(zip(self.stderr_observed_epoch_us, self.stderr_lines))
        result, allocated = [], {}
        for observed_epoch_us, line in rows:
            match = re.fullmatch(r"\S+\s+I\s+slot\s+launch_slot_:\s+id\s+([0-9]+)\s+\|\s+"
                                 r"task\s+([0-9]+)\s+\|\s+processing task, is_child = 0", line)
            if match:
                allocated[int(match[2])] = {"slot_id": int(match[1]), "task_id": int(match[2]),
                                           "observed_epoch_us": observed_epoch_us, "line": line}
            prompt = re.fullmatch(r"\S+\s+I\s+slot\s+operator\(\):\s+id\s+([0-9]+)\s+\|\s+"
                                  r"task\s+([0-9]+)\s+\|\s+new prompt, .*task.n_tokens = ([0-9]+)", line)
            if prompt and int(prompt[2]) in allocated:
                event = allocated.pop(int(prompt[2]))
                if event["slot_id"] == int(prompt[1]):
                    result.append({**event, "prompt_tokens": int(prompt[3]), "prefill_line": line,
                                   "prefill_observed_epoch_us": observed_epoch_us})
        return tuple(result)

    def ffn_call_events(self) -> tuple[dict[str, object], ...]:
        """Return timestamped physical FFN calls observed on stderr."""

        with self._stderr_lock:
            rows = tuple(zip(
                self.stderr_observed_epoch_us,
                self.stderr_lines,
            ))
        result = []
        for line_index, (observed_epoch_us, line) in enumerate(rows):
            call = parse_llama_server_ffn_call(line)
            if call is None:
                continue
            result.append({
                "columns": call.columns,
                "contexts": [
                    {
                        "plan_generation": row.plan_generation,
                        "rows": row.rows,
                        "scheduler_request_id": row.scheduler_request_id,
                        "server_slot_id": row.server_slot_id,
                    }
                    for row in call.contexts
                ],
                "layer": call.layer,
                "line_index": line_index,
                "observed_epoch_us": observed_epoch_us,
                "payload_bytes": call.payload_bytes,
                "request_id": call.request_id,
                "tokens": call.tokens,
            })
        return tuple(result)

    def begin_execution(
        self,
        command: PhysicalExecutionCommand,
        manifest: ModelManifest,
    ) -> LlamaServerExecutionMarker:
        if not isinstance(command, PhysicalExecutionCommand):
            raise PhysicalAdapterError(
                "llama-server execution command is invalid"
            )
        validate_physical_execution_command(command)
        contract = llama_server_launch_contract(command, manifest)
        if not _launch_contract_supports_execution(
            self.launch_contract, contract
        ):
            raise PhysicalAdapterError(
                "llama-server execution differs from the launched contract"
            )
        if self.process is None or self.process.poll() is not None:
            raise LlamaServerExitedError(
                "llama-server execution endpoint is not active",
                executor_id=command.executor_id,
                returncode=self.exit_code(),
            )
        phone_contract = None
        phone_source = _phone_ffn_command(command)
        if (
            contract.phone_device_id is not None
            and (
                phone_source.execution_contract.execution_mode != "desktop"
                or phone_source.execution_contract.remote_resident_ffn is not None
            )
        ):
            phone_contract = (
                phone_ffn_resident_contract(command, manifest)
                if _adaptive_phone_ffn(command)
                else phone_ffn_execution_contract(command, manifest)
            )
        with self._stderr_lock:
            stderr_index = len(self.stderr_lines)
        return LlamaServerExecutionMarker(
            ticket_id=command.ticket_id,
            artifact_sha256=command.artifact_sha256,
            operator_plan_sha256=command.operator_plan_sha256,
            executor_id=command.executor_id,
            stderr_index=stderr_index,
            phone_contract=phone_contract,
            remote_resident_identity_sha256=(
                None if command.execution_contract.remote_resident_ffn is None
                else canonical_sha256(command.execution_contract.remote_resident_ffn.to_json())
            ),
        )

    def _drain_execution_calls(
        self,
        marker: LlamaServerExecutionMarker,
        expected_scope_ids: set[str],
        lines: tuple[str, ...],
        calls: tuple[LlamaServerFfnCall, ...],
        expected_calls: tuple[tuple[int, int], ...] | None,
    ) -> tuple[tuple[str, ...], tuple[LlamaServerFfnCall, ...]]:
        if expected_calls is None:
            return lines, calls
        deadline = time.monotonic() + _FFN_PROOF_DRAIN_TIMEOUT_S
        with self._stderr_lock:
            while sum(
                self._logical_call_rows(call, expected_scope_ids)
                for call in calls
            ) < len(expected_calls):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._stderr_lock.wait(remaining)
                lines, calls = self._execution_stderr_calls(
                    marker, expected_scope_ids
                )
        return lines, calls

    def finish_execution(
        self,
        marker: LlamaServerExecutionMarker,
        command: PhysicalExecutionCommand,
        manifest: ModelManifest,
        *,
        output_tokens: int,
        adaptive_observation: AdaptiveDecodeGroupedObservation | None = None,
        static_control_ack: Mapping[str, object] | None = None,
        helper_envelopes: Sequence[
            RuntimeHelperExecutionEnvelope
        ] = (),
    ) -> LlamaServerExecutionProof:
        self._validate_execution_finish(marker, command, output_tokens)
        (
            phone_source,
            helper_rows,
            current_helper,
            contract_by_plan,
            expected_contract,
        ) = self._execution_phone_contracts(
            marker, command, manifest, helper_envelopes
        )
        static_policy, static_applied_token, static_slot_id = (
            self._static_execution_ack(
                command, output_tokens, static_control_ack
            )
        )
        cohort = command.decode_cohort
        cohort_leader = (
            None if cohort is None
            else cohort.get("leader_request_id")
        )
        expected_scope_ids = (
            {command.request_id}
            if cohort is None
            else set(cohort.get("member_request_ids", ()))
        )
        with self._stderr_lock:
            lines, calls = self._execution_stderr_calls(
                marker, expected_scope_ids
            )
        adaptive = _adaptive_phone_ffn(command)
        if (
            (not adaptive and adaptive_observation is not None)
            or (
                adaptive
                and cohort is None
                and adaptive_observation is None
            )
        ):
            raise PhysicalAdapterError(
                "adaptive execution proof is absent or unexpected"
            )
        expected_calls, stale_tail = self._adaptive_expected_calls(
            adaptive_observation,
            command,
            phone_source,
            contract_by_plan,
            cohort,
            cohort_leader,
            output_tokens,
        )
        if stale_tail:
            calls = ()
        lines, calls = self._drain_execution_calls(
            marker, expected_scope_ids, lines, calls, expected_calls
        )
        by_layer, call_plan_by_request = self._verify_execution_calls(
            calls=calls,
            expected_contract=expected_contract,
            expected_calls=expected_calls,
            observation=adaptive_observation,
            command=command,
            manifest=manifest,
            contract_by_plan=contract_by_plan,
            expected_scope_ids=expected_scope_ids,
            cohort=cohort,
            adaptive=adaptive,
            phone_source=phone_source,
            current_helper=current_helper,
            output_tokens=output_tokens,
            static_applied_token=static_applied_token,
        )
        session_proofs = self._execution_session_proofs(
            command,
            phone_source,
            helper_rows,
            adaptive_observation,
            calls,
            call_plan_by_request,
            manifest,
            expected_scope_ids,
        )
        return self._execution_proof(
            marker=marker,
            command=command,
            lines=lines,
            calls=calls,
            by_layer=by_layer,
            session_proofs=session_proofs,
            observation=adaptive_observation,
            static_policy=static_policy,
            static_applied_token=static_applied_token,
            static_slot_id=static_slot_id,
            manifest=manifest,
            expected_scope_ids=expected_scope_ids,
        )

    def bind_ready_helper(
        self,
        marker: LlamaServerExecutionMarker,
        command: PhysicalExecutionCommand,
        manifest: ModelManifest,
    ) -> LlamaServerExecutionMarker:
        """Bind a READY late helper to an existing desktop marker."""

        if (
            not isinstance(marker, LlamaServerExecutionMarker)
            or not isinstance(command, PhysicalExecutionCommand)
            or command.helper_envelope is None
            or marker.phone_contract is not None
            or dormant_phone_ffn_parameters(
                command.adapter_parameters
            ) is None
            or (
                marker.ticket_id,
                marker.artifact_sha256,
                marker.operator_plan_sha256,
                marker.executor_id,
            ) != (
                command.ticket_id,
                command.artifact_sha256,
                command.operator_plan_sha256,
                command.executor_id,
            )
        ):
            raise PhysicalAdapterError(
                "late phone helper differs from the execution marker"
            )
        validate_physical_execution_command(command)
        if (
            not _launch_contract_supports_execution(
                self.launch_contract,
                llama_server_launch_contract(command, manifest),
            )
            or not _adaptive_phone_ffn(command)
        ):
            raise PhysicalAdapterError(
                "late phone helper differs from the launched contract"
            )
        return replace(
            marker,
            phone_contract=phone_ffn_resident_contract(
                command, manifest
            ),
        )

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        for stream in (self._stderr_file, self._stdout_file):
            if stream is not None:
                stream.close()


class LlamaServerProcessLauncher:
    """Launch only the endpoint and placement encoded in a ticket."""

    _SPLIT_PREFIXES = ("S41_SERVER_FFN_", "LLAMA_FFN_SPLIT_")

    def __init__(self, configuration: LlamaServerProcessConfiguration) -> None:
        if not isinstance(configuration, LlamaServerProcessConfiguration):
            raise PhysicalAdapterError(
                "llama-server process launcher configuration is invalid"
            )
        self._configuration = configuration
        # file digests keyed by (path, size, mtime_ns): the speculative draft and the server pin
        self._digest_cache: dict[tuple[str, int, int], str] = {}

    @staticmethod
    def _endpoint(endpoint: str) -> tuple[str, int]:
        parsed = urlsplit(endpoint)
        if not (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and parsed.port is not None
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
        ):
            raise PhysicalAdapterError(
                "managed llama-server endpoint is not local"
            )
        return parsed.hostname, parsed.port

    @staticmethod
    def _healthy(host: str, port: int) -> bool:
        connection = http.client.HTTPConnection(host, port, timeout=1)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            payload = json.loads(response.read())
            return (
                response.status == 200
                and type(payload) is dict
                and payload.get("status") == "ok"
            )
        finally:
            connection.close()

    def _file_sha256(self, path: Path) -> str:
        """Digest of a file, computed once per (path, size, mtime)."""
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        digest = self._digest_cache.get(key)
        if digest is None:
            digest = self._digest_cache[key] = _sha256_file(path)
        return digest

    def _speculative_arguments(
        self, contract: LlamaServerLaunchContract, device_name: str | None
    ) -> list[str]:
        """Draft-model flags of a launch that speculates (campaign ``speculative_rows``).

        The draft file must match its contract by size and digest and a phone launch must run
        the pinned patched server (the stock server leaks its FFN split into the draft); the
        fork needs ``--spec-type draft-simple`` next to the draft model. A draft on the GPU
        goes to the parent's device; a CPU parent keeps the draft on the CPU."""
        speculative = contract.speculative
        path = Path(speculative.draft_model_path)
        if (
            not path.is_file()
            or path.stat().st_size != speculative.draft_bytes
            or self._file_sha256(path) != speculative.draft_sha256
        ):
            raise PhysicalAdapterError("llama-server speculative draft differs from its contract")
        if speculative.patched_server_sha256 is not None and (
            self._file_sha256(self._configuration.server_path) != speculative.patched_server_sha256
        ):
            raise PhysicalAdapterError("llama-server binary differs from the speculative patched server pin")
        if speculative.patched_server_sha256 is None and contract.phone_device_id is not None:
            raise PhysicalAdapterError("llama-server phone launch cannot draft without the patched server pin")
        arguments = [
            "--spec-type", "draft-simple",
            "--spec-draft-model", str(path),
            "--spec-draft-n-max", str(speculative.draft_max),
            "--spec-draft-n-min", str(speculative.draft_min),
            "--spec-draft-ngl", str(speculative.draft_gpu_layers),
        ]
        if speculative.draft_gpu_layers > 0:
            if contract.gpu_layers == 0 or device_name is None:
                raise PhysicalAdapterError("llama-server draft GPU layers require a GPU parent")
            arguments.extend(("--spec-draft-device", device_name))
        return arguments

    def _launch_contract(
        self,
        endpoint: str,
        contract: LlamaServerLaunchContract,
        manifest: ModelManifest,
        *,
        label: str,
        control_check: Callable[[], None],
    ) -> ManagedLlamaServer:
        if not callable(control_check):
            raise PhysicalAdapterError(
                "llama-server process control check is invalid"
            )
        model_path = self._configuration.model_paths_by_artifact.get(
            manifest.artifact_sha256
        )
        if model_path is None:
            raise PhysicalAdapterError(
                "llama-server model artifact is not configured"
            )
        host, port = self._endpoint(endpoint)
        arguments = [
            str(self._configuration.server_path),
            "--model", str(model_path),
            "--alias", contract.model_alias,
            "--fit", "off",
            "--ctx-size", str(contract.context_size),
            "--parallel", str(contract.parallel),
            "--batch-size", str(contract.batch_size),
            "--ubatch-size", str(contract.ubatch_size),
            "--flash-attn", "on",
            "--cont-batching",
            "--kv-unified",
            "--no-cache-idle-slots",
            "--cache-type-k", "f16",
            "--cache-type-v", "f16",
            "--split-mode", "none",
            "--n-gpu-layers", str(contract.gpu_layers),
            "--main-gpu", "0",
            "--host", host,
            "--port", str(port),
            "--metrics",
            "--slots",
            "--no-webui",
            "--log-colors", "off",
            "--log-timestamps",
            "--verbose",
        ]
        device_name = self._configuration.executable_device_names.get(
            contract.gpu_device_id
        )
        if contract.desktop_launch_mode == "runtime-defaults":
            defaults = {
                "--fit": 1, "--flash-attn": 1, "--cont-batching": 0,
                "--kv-unified": 0, "--no-cache-idle-slots": 0,
                "--cache-type-k": 1, "--cache-type-v": 1,
                "--split-mode": 1, "--main-gpu": 1,
            }
            resolved = []
            source_arguments = iter(arguments)
            for argument in source_arguments:
                if argument not in defaults:
                    resolved.append(argument)
                elif defaults[argument]:
                    next(source_arguments)
            arguments = resolved
        if contract.gpu_layers > 0:
            if device_name is None:
                raise PhysicalAdapterError(
                    "llama-server GPU device mapping is absent"
                )
            arguments.extend(("--device", device_name))
        else:
            arguments.extend(("--device", "none", "--no-kv-offload"))
        if contract.threads:
            arguments.extend((
                "--threads", str(contract.threads),
                "--threads-batch", str(contract.threads_batch),
            ))
        if contract.kv_cpu_layers:
            if any(layer >= manifest.block_count for layer in contract.kv_cpu_layers):
                raise PhysicalAdapterError("llama-server CPU KV layers exceed the model")
            arguments.extend(("--kv-cpu-layers", ",".join(map(str, contract.kv_cpu_layers))))
        if contract.kv_device_cells:
            if any(layer >= manifest.block_count for layer, _ in contract.kv_device_cells):
                raise PhysicalAdapterError("llama-server split KV layers exceed the model")
            arguments.extend(("--kv-device-cells", ",".join(f"{layer}:{cells}" for layer, cells in contract.kv_device_cells)))
        if contract.speculative is not None:
            arguments.extend(self._speculative_arguments(contract, device_name))
        if contract.cpu_affinity is not None:
            arguments = [
                "taskset", "--cpu-list", contract.cpu_affinity, *arguments
            ]
        environment = os.environ.copy()
        if contract.gpu_layers == 0:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        # Native graph mode is presence-based and cached for the process lifetime.
        environment.pop("GGML_CUDA_DISABLE_GRAPHS", None)
        if contract.cuda_graph_mode == "disabled":
            environment["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        environment.pop("S41_SERVER_LOGITS_TRACE", None)
        if contract.logits_trace_path is not None:
            environment["S41_SERVER_LOGITS_TRACE"] = contract.logits_trace_path
        environment.pop("GGML_SCHED_TRACE", None)
        if contract.scheduler_trace_path is not None:
            environment["GGML_SCHED_TRACE"] = contract.scheduler_trace_path
        for name in tuple(environment):
            if name.startswith(self._SPLIT_PREFIXES):
                del environment[name]
        environment.update(contract.ffn_environment)
        libraries = tuple(dict.fromkeys((
            *self._configuration.common_library_paths,
            *self._configuration.library_paths_by_device.get(
                contract.gpu_device_id, ()
            ),
        )))
        if libraries:
            environment["LD_LIBRARY_PATH"] = ":".join((
                *(str(path) for path in libraries),
                str(self._configuration.server_path.parent),
                environment.get("LD_LIBRARY_PATH", ""),
            ))
        process = ManagedLlamaServer(
            tuple(arguments),
            environment,
            self._configuration.output_directory,
            label,
            contract,
        )
        process.start()
        deadline = time.monotonic() + 300
        try:
            while time.monotonic() < deadline:
                control_check()
                if process.process is None or process.process.poll() is not None:
                    raise PhysicalAdapterError(
                        "llama-server exited during model load"
                    )
                try:
                    if self._healthy(host, port):
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.1)
            else:
                raise PhysicalAdapterError(
                    "llama-server model load timed out"
                )
            def confirmed(marker: str) -> bool:
                """Wait for a startup marker instead of sampling once.

                The server prints these before it serves traffic, but the reader thread that fills
                `stderr_lines` can lag the health endpoint under load, and a one-shot check then fails a
                server that is correct. Measured 2026-09-22: a Qwen reload logged `dormant_policy` 72 s
                into startup and the launcher missed it, aborting the arm at 25 of 31 streams.
                """
                limit = time.monotonic() + _STARTUP_MARKER_TIMEOUT_S
                while True:
                    if marker in "\n".join(process.stderr_lines):
                        return True
                    if time.monotonic() >= limit:
                        return False
                    control_check()
                    time.sleep(0.05)

            if contract.logits_trace_path is not None:
                marker = "S41SERVERLOGITS schema=s41-logits-v1 path=" + contract.logits_trace_path
                if not confirmed(marker):
                    raise PhysicalAdapterError("llama-server did not confirm raw logits tracing support")
            if contract.ffn_row_diagnostic_steps:
                marker = f"S41SERVERFFN row_diagnostic_steps={contract.ffn_row_diagnostic_steps} local_shadow=1 delayed_release=1"
                if not confirmed(marker):
                    raise PhysicalAdapterError("llama-server did not confirm the FFN row diagnostic contract")
            if contract.scheduler_trace_path is not None:
                marker = "GGML_SCHED_TRACE schema=ggml-sched-trace-v1 path=" + contract.scheduler_trace_path
                if not confirmed(marker):
                    raise PhysicalAdapterError("llama-server did not confirm scheduler tracing support")
            if (contract.ffn_environment.get("S41_SERVER_FFN_DORMANT_HOST_SHARE") == "1"
                    and (contract.ffn_host_share_drop_cache != 1 or contract.ffn_host_share_populate != 1)):
                # Only a server that runs the dormant host share logs its policy; a plain server has none.
                policy = (f"S41SERVERFFN dormant_policy drop_cache={contract.ffn_host_share_drop_cache} "
                          f"populate={contract.ffn_host_share_populate}")
                if not confirmed(policy):
                    raise PhysicalAdapterError("llama-server did not confirm the dormant host share policy")
            log = "\n".join(process.stderr_lines)
            placements = re.findall(
                r"offloaded ([0-9]+)/([0-9]+) layers to GPU", log
            )
            if contract.gpu_layers > 0 and not any(
                int(loaded) == contract.gpu_layers and int(total) > 0
                for loaded, total in placements
            ):
                raise PhysicalAdapterError(
                    "llama-server GPU placement differs from the ticket"
                )
            split_ready = sum(
                line.startswith("S41SERVERFFN ready ")
                for line in process.stderr_lines
            )
            if split_ready != int(bool(contract.ffn_environment)):
                raise PhysicalAdapterError(
                    "llama-server split readiness differs from the ticket"
                )
            expected_remote_mask = contract.ffn_environment.get(
                "S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK"
            )
            proof_reader = getattr(process, "remote_resident_proof", None)
            proof = None if proof_reader is None else proof_reader()
            if expected_remote_mask is None:
                if proof is not None:
                    raise PhysicalAdapterError(
                        "llama-server omitted weights without a remote-resident ticket"
                    )
            elif (
                proof is None
                or str(proof.layer_mask) != expected_remote_mask
                or proof.warmup != "validated"
            ):
                raise PhysicalAdapterError(
                    "llama-server remote-resident proof is absent or differs"
                )
            return process
        except BaseException:
            process.stop()
            raise

    def launch(
        self,
        command: PhysicalExecutionCommand | PhysicalTransitionCommand,
        manifest: ModelManifest,
        *,
        label: str,
        control_check: Callable[[], None],
    ) -> ManagedLlamaServer:
        contract = llama_server_launch_contract(command, manifest)
        endpoint = (
            command.participant.endpoint
            if isinstance(command, PhysicalTransitionCommand)
            else command.endpoint
        )
        return self._launch_contract(
            endpoint,
            contract,
            manifest,
            label=label,
            control_check=control_check,
        )

    def launch_contract(
        self,
        endpoint: str,
        contract: LlamaServerLaunchContract,
        manifest: ModelManifest,
        *,
        label: str,
        control_check: Callable[[], None],
    ) -> ManagedLlamaServer:
        """Launch an exact scheduler-produced contract for calibration."""
        if not isinstance(contract, LlamaServerLaunchContract):
            raise PhysicalAdapterError(
                "llama-server calibration contract is invalid"
            )
        return self._launch_contract(
            endpoint,
            contract,
            manifest,
            label=label,
            control_check=control_check,
        )

    def launch_capability(
        self,
        capability: RuntimeExecutorCapability,
        manifest: ModelManifest,
        *,
        label: str,
        control_check: Callable[[], None],
    ) -> ManagedLlamaServer:
        contract = llama_server_capability_contract(capability, manifest)
        return self._launch_contract(
            capability.endpoint,
            contract,
            manifest,
            label=label,
            control_check=control_check,
        )
