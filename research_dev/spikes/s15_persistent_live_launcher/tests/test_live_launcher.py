#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
PERSISTENT = SPIKE.parent / "s15_persistent_runtime"
S15 = SPIKE.parent / "s15_runtime_dispatch"
S14 = SPIKE.parent / "s14_mixed_streaming_scheduler"
sys.path[:0] = [str(SPIKE), str(PERSISTENT), str(S15), str(S14)]

from executor_contract import BOUNDARY_SCHEMA, ExecutionRequest, ExecutorError  # noqa: E402
from live_adapter import (  # noqa: E402
    LiveAdapterError,
    PersistentLiveExecutor,
    PersistentLiveSessionTransport,
)
from live_contract import (  # noqa: E402
    FrozenRoute,
    LaunchSpec,
    LiveContractError,
    canonical,
    digest,
    execute_record,
    read_artifact,
    safe_artifact,
    strict_line,
)
from persistent_bridge import CONFIG_SCHEMA, load_config  # noqa: E402
from persistent_transport import PersistentPreparedTransport  # noqa: E402
from physical_executor import PhysicalRouteBinding, TransportReply  # noqa: E402
from session_adapter import (  # noqa: E402
    PersistentWorkerCapability,
    StageNetSessionAdapter,
    StageNetSessionBinding,
)


PROFILE = "sha256:" + "ab" * 32
EVIDENCE = "sha256:" + "bc" * 32
BINARY = "sha256:" + "cd" * 32
COHORT = "sha256:" + "de" * 32
MANIFEST = "sha256:" + "ef" * 32
BOOT = "boot-op15"


def route(batch: int = 2) -> FrozenRoute:
    return FrozenRoute(
        "op15-head-0-8", PROFILE, EVIDENCE, "op15:serial", BOOT, BINARY,
        (0, 8), (8, 48), 14, 1, 1, 1, batch, 8,
    )


def tokens(launch_id: int, batch: int, n_gen: int) -> list[list[int]]:
    return [
        [1000 + launch_id * 100 + stream * 10 + index for index in range(n_gen)]
        for stream in range(batch)
    ]


def spec(launch_id: int, ending: str, batch: int = 2, n_gen: int = 3,
         expected: str | None = None) -> LaunchSpec:
    token_digest = digest(canonical({"token_ids": tokens(launch_id, batch, n_gen)}))
    return LaunchSpec(
        launch_id, "Explain batching.", n_gen,
        tuple(f"l{launch_id}-r{index}" for index in range(batch)),
        ending, 500_000, token_digest if expected is None else expected,
    )


def physical_binding(allowed_cpu_ops=("GET_ROWS",)) -> PhysicalRouteBinding:
    return PhysicalRouteBinding(
        "op15-head-0-8", PROFILE, "op15:serial", 1, BINARY, 2, 1, BOOT,
        COHORT, MANIFEST, (0, 8), "HTP0", tuple(allowed_cpu_ops),
    )


def execution_request(value: LaunchSpec) -> ExecutionRequest:
    return ExecutionRequest(
        value.launch_id, "op15-head-0-8", PROFILE, "op15:serial",
        14, 1, value.launch_id, 1, 1, "gemma|decode|head-0-8",
        value.request_ids, COHORT, MANIFEST, value.deadline_us, BOUNDARY_SCHEMA,
    )


def route_record(value: FrozenRoute) -> dict:
    return {
        "route_id": value.route_id,
        "profile_id": value.profile_id,
        "evidence_sha256": value.evidence_sha256,
        "device_id": value.device_id,
        "device_boot_id": value.device_boot_id,
        "worker_binary_sha256": value.worker_binary_sha256,
        "layer_range": list(value.layer_range),
        "host_tail_range": list(value.host_tail_range),
        "route_epoch": value.route_epoch,
        "residency_epoch": value.residency_epoch,
        "device_boot_epoch": value.device_boot_epoch,
        "registry_generation": value.registry_generation,
        "batch_size": value.batch_size,
        "max_n_gen": value.max_n_gen,
    }


class Stack:
    def __init__(self, root: Path, mode: str, specs: tuple[LaunchSpec, ...],
                 allowed_cpu_ops=("GET_ROWS",)) -> None:
        self.route = route(len(specs[0].request_ids))
        self.child_artifacts = root / "child-artifacts"
        child = [
            sys.executable, str(HERE / "fixture_child.py"), "--mode", mode,
            "--boot", BOOT, "--layer-start", "0", "--layer-end", "8",
        ]
        config = {
            "schema": CONFIG_SCHEMA,
            "route": route_record(self.route),
            "child_command": child,
            "artifact_root": str(self.child_artifacts),
            "child_ready_timeout_s": 3,
        }
        config_path = root / "bridge.config.json"
        config_path.write_bytes(canonical(config))
        self.outer = PersistentPreparedTransport(
            (sys.executable, str(SPIKE / "persistent_bridge.py"),
             "--config", str(config_path)),
            root / "outer-artifacts", ready_timeout_s=3,
        )
        binding = physical_binding(allowed_cpu_ops)
        stage = StageNetSessionAdapter(
            StageNetSessionBinding(BINARY, 1, BOOT, (0, 8), 48),
            (PersistentWorkerCapability(BINARY, 2, True),),
        )
        self.session = stage
        self.live_transport = PersistentLiveSessionTransport(
            self.outer, self.route, binding, stage,
            {item.launch_id: item for item in specs}, self.child_artifacts,
        )
        self.executor = PersistentLiveExecutor(binding, self.live_transport)

    def close(self) -> None:
        if self.outer.state != "FINALIZED":
            self.outer.terminate()
            self.outer.finalize()


class ContractTests(unittest.TestCase):
    def test_execute_record_is_canonical_and_binds_every_identity(self) -> None:
        rt = route()
        item = spec(1, "DETACH")
        value = execute_record(rt, item, 1, COHORT, MANIFEST)
        self.assertEqual(strict_line(canonical(value), "execute"), value)
        self.assertEqual(
            set(value),
            {
                "schema", "command", "protocol_version", "launch_id", "prompt",
                "n_gen", "batch_size", "request_ids", "session_end", "deadline_us",
                "expected_route_id", "expected_profile_id", "expected_evidence_sha256",
                "expected_device_id", "expected_device_boot_id",
                "expected_worker_binary_sha256", "expected_layer_range", "route_epoch",
                "expected_host_tail_range",
                "residency_epoch", "lease_epoch", "device_boot_epoch",
                "registry_generation", "cohort_sha256", "input_manifest_sha256",
            },
        )

    def test_duplicate_noncanonical_and_path_escape_are_rejected(self) -> None:
        with self.assertRaises(LiveContractError):
            strict_line(b'{"a":1,"a":2}\n', "duplicate")
        with self.assertRaises(LiveContractError):
            strict_line(b'{"a": 1}\n', "noncanonical")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(LiveContractError):
                safe_artifact(Path(tmp), "../escape")

    def test_symlink_artifact_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            target = Path(tmp) / "target.bin"
            target.write_bytes(b"payload")
            (root / "alias.bin").symlink_to(target)
            with self.assertRaisesRegex(LiveContractError, "symlink"):
                read_artifact(root, "alias.bin")

    def test_child_readiness_timeout_is_typed_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, bad in enumerate((True, 0, 601)):
                value = {
                    "schema": CONFIG_SCHEMA,
                    "route": route_record(route()),
                    "child_command": ["unused"],
                    "artifact_root": str(root / f"artifacts-{index}"),
                    "child_ready_timeout_s": bad,
                }
                path = root / f"bad-{index}.json"
                path.write_bytes(canonical(value))
                with self.subTest(value=bad), self.assertRaises(LiveContractError):
                    load_config(path)
            valid = {
                "schema": CONFIG_SCHEMA,
                "route": route_record(route()),
                "child_command": ["unused"],
                "artifact_root": str(root / "valid-artifacts"),
                "child_ready_timeout_s": 240,
            }
            path = root / "valid.json"
            path.write_bytes(canonical(valid))
            self.assertEqual(load_config(path)[3], 240)


class LiveIntegrationTests(unittest.TestCase):
    def test_detach_then_stop_passes_one_child_and_exact_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            specs = (spec(1, "DETACH"), spec(2, "STOP"))
            stack = Stack(Path(tmp), "normal", specs)
            try:
                first = stack.executor.launch(execution_request(specs[0]), 0)
                second = stack.executor.launch(execution_request(specs[1]), first.finish_us)
                self.assertEqual((first.outcome, second.outcome), ("completed", "completed"))
                self.assertEqual(stack.session.state, "STOPPED")
                self.assertIsNone(stack.live_transport.pending_detach)
                command = strict_line(
                    (stack.child_artifacts / "launch-000001/child_command.bin").read_bytes(),
                    "child command",
                )
                self.assertEqual(set(command), {
                    "schema", "launch_id", "prompt", "n_gen", "request_count", "session_end",
                })
                self.assertEqual(command["request_count"], 2)
                child = (stack.child_artifacts / "launch-000001/child_result.bin")
                raw_child = child.read_bytes()
                self.assertIn(b'"route_wall_us":801', raw_child)
                self.assertLess(
                    raw_child.index(b'"elapsed_us"'),
                    raw_child.index(b'"route_wall_us"'),
                )
                self.assertTrue(
                    (stack.child_artifacts / "launch-000001/host_placement.bin").is_file()
                )
            finally:
                stack.close()

    def test_expected_token_digest_is_load_bearing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = spec(1, "DETACH", expected="sha256:" + "00" * 32)
            stack = Stack(Path(tmp), "normal", (bad,))
            try:
                with self.assertRaisesRegex(ExecutorError, "token artifact digest"):
                    stack.executor.launch(execution_request(bad), 0)
                self.assertEqual(stack.session.state, "POISONED")
            finally:
                stack.close()

    def test_cpp_encoded_command_bound_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item = spec(1, "DETACH")
            object.__setattr__(item, "prompt", "\x01" * (16 * 1024))
            stack = Stack(Path(tmp), "normal", (item,))
            try:
                result = stack.executor.launch(execution_request(item), 0)
                self.assertNotEqual(result.outcome, "completed")
                self.assertEqual(stack.session.state, "POISONED")
            finally:
                stack.close()

    def test_child_command_artifact_is_load_bearing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item = spec(1, "DETACH")
            stack = Stack(Path(tmp), "normal", (item,))
            original = stack.outer.exchange

            def tamper(request, payload):
                reply = original(request, payload)
                result = strict_line(reply.payload, "bridge result")
                path = stack.child_artifacts / "launch-000001/child_command.bin"
                command = strict_line(path.read_bytes(), "child command")
                command["prompt"] = "a different prompt"
                changed = canonical(command)
                path.write_bytes(changed)
                result["artifact_hashes"]["child_command"]["sha256"] = digest(changed)
                return TransportReply("reply", reply.elapsed_us, canonical(result))

            stack.outer.exchange = tamper
            try:
                with self.assertRaisesRegex(ExecutorError, "child command artifact"):
                    stack.executor.launch(execution_request(item), 0)
                self.assertEqual(stack.session.state, "POISONED")
            finally:
                stack.close()

    def test_artifacts_must_belong_to_the_active_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item = spec(1, "DETACH")
            stack = Stack(Path(tmp), "normal", (item,))
            original = stack.outer.exchange

            def tamper(request, payload):
                reply = original(request, payload)
                result = strict_line(reply.payload, "bridge result")
                result["artifact_hashes"]["child_command"] = dict(
                    result["artifact_hashes"]["tokens"]
                )
                return TransportReply("reply", reply.elapsed_us, canonical(result))

            stack.outer.exchange = tamper
            try:
                with self.assertRaisesRegex(ExecutorError, "not launch-specific"):
                    stack.executor.launch(execution_request(item), 0)
                self.assertEqual(stack.session.state, "POISONED")
            finally:
                stack.close()

    def test_nested_numeric_type_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item = spec(1, "DETACH")
            stack = Stack(Path(tmp), "normal", (item,))
            original = stack.outer.exchange

            def tamper(request, payload):
                reply = original(request, payload)
                result = strict_line(reply.payload, "bridge result")
                result["identity"]["route_epoch"] = float(
                    result["identity"]["route_epoch"]
                )
                return TransportReply("reply", reply.elapsed_us, canonical(result))

            stack.outer.exchange = tamper
            try:
                with self.assertRaisesRegex(ExecutorError, "frozen identity"):
                    stack.executor.launch(execution_request(item), 0)
                self.assertEqual(stack.session.state, "POISONED")
            finally:
                stack.close()

    def test_host_placement_failures_never_complete(self) -> None:
        modes = (
            "missing_host_placement", "duplicate_host_placement", "host_cpu",
            "host_wrong_backend", "host_range",
        )
        for mode in modes:
            with tempfile.TemporaryDirectory() as tmp:
                item = spec(1, "DETACH")
                stack = Stack(Path(tmp), mode, (item,))
                try:
                    try:
                        result = stack.executor.launch(execution_request(item), 0)
                    except ExecutorError:
                        result = None
                    with self.subTest(mode=mode):
                        self.assertTrue(result is None or result.outcome != "completed")
                        self.assertEqual(stack.session.state, "POISONED")
                finally:
                    stack.close()

    def test_host_placement_artifact_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item = spec(1, "DETACH")
            stack = Stack(Path(tmp), "normal", (item,))
            original = stack.outer.exchange

            def tamper(request, payload):
                reply = original(request, payload)
                result = strict_line(reply.payload, "bridge result")
                path = stack.child_artifacts / "launch-000001/host_placement.bin"
                line = path.read_bytes()
                value = json.loads(line[len(b"PLACEMENTCERT "):])
                value["layer_start"] = 9
                changed = b"PLACEMENTCERT " + canonical(value)
                path.write_bytes(changed)
                result["artifact_hashes"]["host_placement"]["sha256"] = digest(changed)
                return TransportReply("reply", reply.elapsed_us, canonical(result))

            stack.outer.exchange = tamper
            try:
                with self.assertRaisesRegex(ExecutorError, "PLACEMENTCERT identity"):
                    stack.executor.launch(execution_request(item), 0)
                self.assertEqual(stack.session.state, "POISONED")
            finally:
                stack.close()

    def test_noncanonical_phone_cert_is_accepted_but_duplicate_key_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item = spec(1, "STOP")
            stack = Stack(Path(tmp), "noncanonical_session", (item,))
            try:
                result = stack.executor.launch(execution_request(item), 0)
                self.assertEqual(result.outcome, "completed")
                self.assertEqual(stack.session.state, "STOPPED")
            finally:
                stack.close()
        with tempfile.TemporaryDirectory() as tmp:
            item = spec(1, "DETACH")
            stack = Stack(Path(tmp), "duplicate_session_key", (item,))
            try:
                try:
                    result = stack.executor.launch(execution_request(item), 0)
                except ExecutorError:
                    result = None
                self.assertTrue(result is None or result.outcome != "completed")
                self.assertEqual(stack.session.state, "POISONED")
            finally:
                stack.close()

    def test_physical_executor_rejection_prevents_detach_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item = spec(1, "DETACH")
            stack = Stack(Path(tmp), "normal", (item,), allowed_cpu_ops=())
            try:
                with self.assertRaisesRegex(ExecutorError, "undeclared backend fallback"):
                    stack.executor.launch(execution_request(item), 0)
                self.assertEqual(stack.session.state, "POISONED")
                self.assertIsNone(stack.live_transport.pending_detach)
            finally:
                stack.close()

    def test_changed_worker_identity_on_second_session_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            specs = (spec(1, "DETACH"), spec(2, "STOP"))
            stack = Stack(Path(tmp), "changed_pid", specs)
            try:
                self.assertEqual(
                    stack.executor.launch(execution_request(specs[0]), 0).outcome,
                    "completed",
                )
                with self.assertRaisesRegex(ExecutorError, "identity changed"):
                    stack.executor.launch(execution_request(specs[1]), 10_000)
                self.assertEqual(stack.session.state, "POISONED")
            finally:
                stack.close()

    def test_changed_child_host_pid_on_second_session_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            specs = (spec(1, "DETACH"), spec(2, "STOP"))
            stack = Stack(Path(tmp), "changed_host_pid", specs)
            try:
                self.assertEqual(
                    stack.executor.launch(execution_request(specs[0]), 0).outcome,
                    "completed",
                )
                second = stack.executor.launch(execution_request(specs[1]), 10_000)
                self.assertNotEqual(second.outcome, "completed")
                self.assertEqual(stack.session.state, "POISONED")
            finally:
                stack.close()

    def test_reset_and_session_gap_fail_closed(self) -> None:
        for mode, message in (("reset_false", "did not reset"), ("cert_gap", "session_id")):
            with tempfile.TemporaryDirectory() as tmp:
                item = spec(1, "DETACH")
                stack = Stack(Path(tmp), mode, (item,))
                try:
                    with self.subTest(mode=mode), self.assertRaisesRegex(ExecutorError, message):
                        stack.executor.launch(execution_request(item), 0)
                    self.assertEqual(stack.session.state, "POISONED")
                finally:
                    stack.close()

    def test_malformed_duplicate_partial_and_marker_failures_never_complete(self) -> None:
        modes = (
            "malformed_result", "wrong_token_count", "duplicate_result",
            "partial_result", "missing_marker", "wrong_marker", "duplicate_marker",
            "trailing_stderr", "trailing_stdout", "unsolicited",
        )
        for mode in modes:
            with tempfile.TemporaryDirectory() as tmp:
                item = spec(1, "DETACH")
                stack = Stack(Path(tmp), mode, (item,))
                try:
                    try:
                        result = stack.executor.launch(execution_request(item), 0)
                    except ExecutorError:
                        result = None
                    with self.subTest(mode=mode):
                        self.assertTrue(result is None or result.outcome != "completed")
                        self.assertEqual(stack.session.state, "POISONED")
                finally:
                    stack.close()

    def test_unknown_launch_and_request_identity_mutation_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item = spec(1, "DETACH")
            stack = Stack(Path(tmp), "normal", (item,))
            try:
                unknown = execution_request(spec(2, "DETACH"))
                with self.assertRaisesRegex(LiveAdapterError, "no frozen"):
                    stack.executor.launch(unknown, 0)
                changed = execution_request(item)
                object.__setattr__(changed, "timeout_us", item.deadline_us + 1)
                with self.assertRaisesRegex(LiveAdapterError, "frozen route/spec"):
                    stack.executor.launch(changed, 0)
            finally:
                stack.close()


if __name__ == "__main__":
    unittest.main()
