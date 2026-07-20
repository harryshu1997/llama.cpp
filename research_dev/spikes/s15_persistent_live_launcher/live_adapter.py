#!/usr/bin/env python3
"""Typed host adapter for the persistent LayerSplit JSONL bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from executor_contract import ExecutionRequest, ExecutionResult, Executor, ExecutorError
from physical_executor import (
    PhysicalExecutor,
    PhysicalRouteBinding,
    SessionTransport,
    TransportReply,
)
from persistent_transport import PersistentPreparedTransport
from session_adapter import StageNetSessionAdapter, parse_session_cert

from live_contract import (
    CHILD_RESULT_SCHEMA,
    RESULT_SCHEMA,
    FrozenRoute,
    LaunchSpec,
    LiveContractError,
    canonical,
    digest,
    execute_record,
    integer,
    parse_host_placement,
    read_artifact,
    sha256_text,
    strict_line,
    text,
)


REQUIRED_ARTIFACTS = {
    "child_command", "child_result", "child_stdout", "child_stderr",
    "session_cert", "host_placement", "tokens", "placement",
}


class LiveAdapterError(ExecutorError):
    pass


class PersistentLiveSessionTransport(SessionTransport):
    """Turn a rich bridge RESULT into an S15 physical session record."""

    def __init__(self, outer: PersistentPreparedTransport,
                 route: FrozenRoute,
                 physical_binding: PhysicalRouteBinding,
                 session_adapter: StageNetSessionAdapter,
                 specs: Mapping[int, LaunchSpec], artifact_root: Path) -> None:
        if type(outer) is not PersistentPreparedTransport:
            raise LiveAdapterError("outer transport must be PersistentPreparedTransport")
        route.validate()
        physical_binding.validate()
        if not isinstance(session_adapter, StageNetSessionAdapter):
            raise LiveAdapterError("session_adapter has the wrong type")
        if not isinstance(specs, Mapping) or not specs:
            raise LiveAdapterError("specs must be a non-empty mapping")
        normalized = {}
        for launch_id, spec in specs.items():
            if type(launch_id) is not int or type(spec) is not LaunchSpec \
                    or launch_id != spec.launch_id or launch_id in normalized:
                raise LiveAdapterError("spec mapping identity is invalid")
            spec.validate(route)
            normalized[launch_id] = spec
        if not isinstance(artifact_root, Path):
            raise LiveAdapterError("artifact_root must be a Path")
        self._validate_route_binding(route, physical_binding)
        self._outer = outer
        self._route = route
        self._physical_binding = physical_binding
        self._session_adapter = session_adapter
        self._specs = normalized
        self._artifact_root = artifact_root
        self._pending_detach: int | None = None
        self._child_host_pid: int | None = None

    @staticmethod
    def _validate_route_binding(route: FrozenRoute,
                                binding: PhysicalRouteBinding) -> None:
        expected = (
            route.route_id, route.profile_id, route.device_id,
            route.device_boot_id, route.worker_binary_sha256, route.layer_range,
        )
        actual = (
            binding.route_id, binding.profile_id, binding.device_id,
            binding.device_boot_id, binding.worker_binary_sha256, binding.layer_range,
        )
        if actual != expected or binding.expected_backend != "HTP0":
            raise LiveAdapterError("frozen route and physical binding differ")

    @property
    def pending_detach(self) -> int | None:
        return self._pending_detach

    def exchange(self, request: ExecutionRequest, physical_payload: bytes) -> TransportReply:
        if self._pending_detach is not None:
            raise LiveAdapterError("previous DETACH has not passed PhysicalExecutor")
        request.validate()
        spec = self._specs.get(request.launch_id)
        if spec is None:
            raise LiveAdapterError("launch has no frozen persistent spec")
        self._validate_request(request, spec)
        physical = strict_line(physical_payload, "physical executor request")
        if physical.get("launch_id") != request.launch_id \
                or physical.get("route_id") != request.route_id:
            raise LiveAdapterError("physical request payload identity failed")
        self._session_adapter.begin(spec.session_end)
        execute = execute_record(
            self._route, spec, request.lease_epoch,
            request.cohort_sha256, request.input_manifest_sha256,
        )
        try:
            reply = self._outer.exchange(request, canonical(execute))
            reply.validate()
            if reply.state != "reply":
                self._session_adapter.abort(f"outer transport {reply.state}")
                return reply
            result = strict_line(reply.payload, "persistent bridge RESULT")
            cert_line, boundaries = self._validate_result(result, execute, spec, reply)
            adapted = self._session_adapter.accept(
                cert_line,
                result["detach_ack"],
                result["worker_terminated"],
                request,
                self._physical_binding,
                boundaries,
            )
        except Exception as exc:
            if self._session_adapter.state not in ("POISONED", "STOPPED"):
                self._session_adapter.abort(str(exc))
            self._outer.terminate()
            if isinstance(exc, ExecutorError):
                raise
            raise LiveAdapterError(f"persistent live adaptation failed: {exc}") from exc
        if spec.session_end == "DETACH":
            self._pending_detach = request.launch_id
        return TransportReply("reply", reply.elapsed_us, adapted)

    def commit_detach(self, launch_id: int) -> None:
        if type(launch_id) is not int or self._pending_detach != launch_id:
            raise LiveAdapterError("DETACH commit does not match the pending launch")
        self._session_adapter.release_after_detach()
        self._pending_detach = None

    def poison_pending(self, reason: str) -> None:
        if self._session_adapter.state not in ("POISONED", "STOPPED"):
            self._session_adapter.abort(text("poison reason", reason))
        self._pending_detach = None
        self._outer.terminate()

    def _validate_request(self, request: ExecutionRequest, spec: LaunchSpec) -> None:
        expected = (
            self._route.route_id, self._route.profile_id, self._route.device_id,
            self._route.route_epoch, self._route.residency_epoch,
            self._route.device_boot_epoch, self._route.registry_generation,
            spec.request_ids, spec.deadline_us,
        )
        actual = (
            request.route_id, request.profile_id, request.device_id,
            request.route_epoch, request.residency_epoch,
            request.device_boot_epoch, request.registry_generation,
            request.request_ids, request.timeout_us,
        )
        if actual != expected:
            raise LiveAdapterError("execution request differs from its frozen route/spec")

    def _validate_result(self, result: dict, execute: dict, spec: LaunchSpec,
                         reply: TransportReply) -> tuple[bytes, list[dict]]:
        required = {
            "schema", "protocol_version", "launch_id", "terminal",
            "request_count", "request_ids", "session_end", "elapsed_us",
            "route_wall_us", "bridge_elapsed_us", "child_host_pid", "token_sha256",
            "detach_ack", "worker_terminated", "identity", "artifact_hashes",
            "error",
        }
        if set(result) != required:
            raise LiveAdapterError("bridge RESULT has missing or unknown fields")
        expected_terminal = "DETACHED" if spec.session_end == "DETACH" else "STOPPED"
        expected = {
            "schema": RESULT_SCHEMA,
            "protocol_version": 1,
            "launch_id": spec.launch_id,
            "terminal": expected_terminal,
            "request_count": self._route.batch_size,
            "request_ids": list(spec.request_ids),
            "session_end": spec.session_end,
            "error": None,
        }
        for key, wanted in expected.items():
            if type(result.get(key)) is not type(wanted) or result.get(key) != wanted:
                raise LiveAdapterError(f"bridge RESULT identity mismatch: {key}")
        expected_identity = {
            key: execute[key] for key in (
                "expected_route_id", "expected_profile_id", "expected_evidence_sha256",
                "expected_device_id", "expected_device_boot_id",
                "expected_worker_binary_sha256", "expected_layer_range", "route_epoch",
                "expected_host_tail_range",
                "residency_epoch", "lease_epoch", "device_boot_epoch",
                "registry_generation", "cohort_sha256", "input_manifest_sha256",
            )
        }
        if type(result["identity"]) is not dict \
                or canonical(result["identity"]) != canonical(expected_identity):
            raise LiveAdapterError("bridge RESULT frozen identity failed")
        integer("elapsed_us", result["elapsed_us"], 1)
        integer("route_wall_us", result["route_wall_us"], 1)
        integer("bridge_elapsed_us", result["bridge_elapsed_us"], 1)
        integer("child_host_pid", result["child_host_pid"], 1)
        if result["bridge_elapsed_us"] > reply.elapsed_us \
                or reply.elapsed_us > spec.deadline_us:
            raise LiveAdapterError("bridge timing exceeds its typed deadline")
        if result["route_wall_us"] > result["elapsed_us"]:
            raise LiveAdapterError("route wall time exceeds child elapsed time")
        sha256_text("token_sha256", result["token_sha256"])
        child_host_pid = integer("child_host_pid", result["child_host_pid"], 1)
        if self._child_host_pid is not None and child_host_pid != self._child_host_pid:
            raise LiveAdapterError("persistent child host PID changed")
        artifact_bytes = self._verify_artifacts(
            result["artifact_hashes"], spec.launch_id,
        )
        expected_command = {
            "schema": "layersplit-persistent-command-v1",
            "launch_id": spec.launch_id,
            "prompt": spec.prompt,
            "n_gen": spec.n_gen,
            "request_count": self._route.batch_size,
            "session_end": spec.session_end,
        }
        command = strict_line(artifact_bytes["child_command"], "child command artifact")
        if command != expected_command:
            raise LiveAdapterError("child command artifact differs from EXECUTE")
        tokens = strict_line(artifact_bytes["tokens"], "tokens artifact")
        if set(tokens) != {"token_ids"} \
                or digest(artifact_bytes["tokens"]) != result["token_sha256"] \
                or result["token_sha256"] != spec.expected_token_sha256:
            raise LiveAdapterError("token artifact digest failed")
        rows = tokens["token_ids"]
        if type(rows) is not list or len(rows) != self._route.batch_size:
            raise LiveAdapterError("token artifact row count failed")
        for row in rows:
            if type(row) is not list or len(row) != spec.n_gen \
                    or any(type(token) is not int or token < 0 for token in row):
                raise LiveAdapterError("token artifact type or shape failed")
        child = strict_line(artifact_bytes["child_result"], "child result artifact")
        child_required = {
            "schema", "launch_id", "outcome", "host_pid", "request_count",
            "batch_size", "n_gen", "session_end", "elapsed_us", "route_wall_us",
            "token_ids",
        }
        expected_child = {
            "schema": CHILD_RESULT_SCHEMA,
            "launch_id": spec.launch_id,
            "outcome": "completed",
            "host_pid": child_host_pid,
            "request_count": self._route.batch_size,
            "batch_size": self._route.batch_size,
            "n_gen": spec.n_gen,
            "session_end": spec.session_end,
            "elapsed_us": result["elapsed_us"],
            "route_wall_us": result["route_wall_us"],
            "token_ids": tokens["token_ids"],
        }
        if artifact_bytes["child_stdout"] != artifact_bytes["child_result"] \
                or set(child) != child_required \
                or canonical(child) != canonical(expected_child):
            raise LiveAdapterError("child result/stdout/token binding failed")
        cert_line = artifact_bytes["session_cert"]
        cert = parse_session_cert(cert_line)
        if artifact_bytes["child_stderr"].splitlines(keepends=True).count(cert_line) != 1:
            raise LiveAdapterError("child stderr does not contain exactly one SESSIONCERT")
        host_placement_line = artifact_bytes["host_placement"]
        parse_host_placement(
            host_placement_line, self._route.host_tail_range, child_host_pid,
        )
        if artifact_bytes["child_stderr"].splitlines(keepends=True).count(
                host_placement_line) != 1:
            raise LiveAdapterError(
                "child stderr does not contain exactly one host PLACEMENTCERT"
            )
        placement = strict_line(artifact_bytes["placement"], "placement artifact")
        expected_placement = {
            "compute_by_op_and_buffer": cert.get("compute_by_op_and_buffer"),
            "missing_buffer_compute_nodes": cert.get("missing_buffer_compute_nodes"),
            "placement_status": cert.get("placement_status"),
        }
        if canonical(placement) != canonical(expected_placement):
            raise LiveAdapterError("placement artifact differs from SESSIONCERT")
        if type(result["worker_terminated"]) is not bool:
            raise LiveAdapterError("worker_terminated must be bool")
        if spec.session_end == "DETACH":
            if type(result["detach_ack"]) is not int \
                    or type(result["detach_ack"]) is bool \
                    or result["detach_ack"] != 0 \
                    or result["worker_terminated"]:
                raise LiveAdapterError("DETACH completion state failed")
        elif result["detach_ack"] is not None or not result["worker_terminated"]:
            raise LiveAdapterError("STOP completion state failed")
        if self._child_host_pid is None:
            self._child_host_pid = child_host_pid
        boundaries = [
            {
                "request_id": request_id,
                "identity_ok": True,
                "epoch_ok": True,
                "correctness_ok": True,
                "d2h_complete": True,
            }
            for request_id in spec.request_ids
        ]
        return cert_line, boundaries

    def _verify_artifacts(self, bindings: object, launch_id: int) -> dict[str, bytes]:
        if type(bindings) is not dict or set(bindings) != REQUIRED_ARTIFACTS:
            raise LiveAdapterError("bridge artifact binding set is incomplete")
        result = {}
        seen_paths = set()
        expected_parent = f"launch-{launch_id:06d}"
        for name in sorted(REQUIRED_ARTIFACTS):
            binding = bindings[name]
            if type(binding) is not dict or set(binding) != {"path", "sha256"}:
                raise LiveAdapterError("bridge artifact binding has invalid fields")
            sha256_text("artifact sha256", binding["sha256"])
            if binding["path"] != f"{expected_parent}/{name}.bin":
                raise LiveAdapterError("bridge artifact path is not launch-specific")
            path, payload = read_artifact(self._artifact_root, binding["path"])
            if path in seen_paths:
                raise LiveAdapterError("two artifact bindings alias one file")
            seen_paths.add(path)
            if digest(payload) != binding["sha256"]:
                raise LiveAdapterError("bridge artifact digest mismatch")
            result[name] = payload
        return result


class PersistentLiveExecutor(Executor):
    """Commit worker reuse only after PhysicalExecutor accepts the adapted record."""

    def __init__(self, binding: PhysicalRouteBinding,
                 transport: PersistentLiveSessionTransport) -> None:
        if type(binding) is not PhysicalRouteBinding \
                or type(transport) is not PersistentLiveSessionTransport:
            raise LiveAdapterError("persistent executor inputs have invalid types")
        self._transport = transport
        self._physical = PhysicalExecutor((binding,), transport)

    def launch(self, request: ExecutionRequest, now_us: int) -> ExecutionResult:
        try:
            result = self._physical.launch(request, now_us)
            result.validate(request)
            if result.outcome != "completed":
                self._transport.poison_pending(f"physical result was {result.outcome}")
                return result
            if self._transport.pending_detach is not None:
                self._transport.commit_detach(request.launch_id)
            return result
        except Exception as exc:
            self._transport.poison_pending(str(exc))
            if isinstance(exc, ExecutorError):
                raise
            raise LiveAdapterError(f"persistent physical execution failed: {exc}") from exc
