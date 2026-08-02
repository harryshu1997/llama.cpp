#!/usr/bin/env python3

import copy
import importlib.util
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
import types
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "cuda_route_capture_v1.py"
SPEC = importlib.util.spec_from_file_location("cuda_route_capture_v1", SOURCE)
cuda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cuda)


def packed(kind, values):
    values = tuple(values)
    return struct.pack("<" + kind * len(values), *values)


def read_exact(connection, size):
    value = bytearray()
    while len(value) < size:
        block = connection.recv(size - len(value))
        if not block:
            raise EOFError
        value.extend(block)
    return bytes(value)


class FakeStage:
    def __init__(self, sticky_cleanup=False):
        client, server = socket.socketpair()
        self.client = cuda.StageClient(client)
        self.server = server
        self.sticky_cleanup = sticky_cleanup
        self.active = set()
        self.calls = []
        self.error = None
        self.thread = threading.Thread(target=self._serve)
        self.thread.start()

    def _serve(self):
        try:
            while True:
                opcode = struct.unpack("<i", read_exact(self.server, 4))[0]
                if opcode == cuda.STAGE_V3_HELLO:
                    self.server.sendall(packed("i", [
                        cuda.STAGE_V3_MAGIC,
                        cuda.STAGE_V3_VERSION,
                        0,
                        cuda.N_LAYER,
                        cuda.N_LAYER,
                        cuda.N_EMBD,
                        cuda.MAX_STREAMS,
                        cuda.N_CTX_SEQ,
                        cuda.N_BATCH,
                        cuda.N_UBATCH,
                        cuda.CAPABILITIES,
                    ]))
                elif opcode == cuda.STAGE_V3_IDENTITY:
                    self.server.sendall(
                        packed("i", [
                            cuda.STAGE_IDENTITY_MAGIC,
                            cuda.STAGE_IDENTITY_VERSION,
                            cuda.FILE_TYPE,
                        ])
                        + bytes.fromhex(cuda.MODEL_SHA256)
                    )
                elif opcode == cuda.STAGE_V3_STATUS:
                    version = struct.unpack("<i", read_exact(self.server, 4))[0]
                    if version != cuda.STAGE_V3_VERSION:
                        raise AssertionError(version)
                    self._send_status()
                elif opcode == cuda.STAGE_V3_BATCH:
                    version, count, flags = struct.unpack(
                        "<3i",
                        read_exact(self.server, 12),
                    )
                    if version != cuda.STAGE_V3_VERSION or flags != 0:
                        raise AssertionError((version, flags))
                    request_ids = struct.unpack(
                        f"<{count}q",
                        read_exact(self.server, count * 8),
                    )
                    epochs = struct.unpack(
                        f"<{count}q",
                        read_exact(self.server, count * 8),
                    )
                    sequences = struct.unpack(
                        f"<{count}i",
                        read_exact(self.server, count * 4),
                    )
                    positions = struct.unpack(
                        f"<{count}i",
                        read_exact(self.server, count * 4),
                    )
                    tokens = struct.unpack(
                        f"<{count}i",
                        read_exact(self.server, count * 4),
                    )
                    rows = list(zip(
                        request_ids,
                        epochs,
                        sequences,
                        positions,
                        tokens,
                    ))
                    self.calls.append(rows)
                    self.active.update(sequences)
                    outputs = [
                        (token + position + sequence + 1) % 100000
                        for _, _, sequence, position, token in rows
                    ]
                    self.server.sendall(
                        packed("i", [0, count, 0])
                        + packed("q", request_ids)
                        + packed("q", epochs)
                        + packed("i", sequences)
                        + packed("i", positions)
                        + packed("i", outputs)
                    )
                elif opcode == cuda.STAGE_V3_SEQ_REMOVE:
                    version, sequence = struct.unpack(
                        "<2i",
                        read_exact(self.server, 8),
                    )
                    read_exact(self.server, 16)
                    if version != cuda.STAGE_V3_VERSION:
                        raise AssertionError(version)
                    if not self.sticky_cleanup:
                        self.active.discard(sequence)
                    self._send_status()
                elif opcode == cuda.STAGE_STOP:
                    return
                else:
                    raise AssertionError(opcode)
        except BaseException as error:
            self.error = error

    def _send_status(self):
        self.server.sendall(packed("i", [
            0,
            cuda.STAGE_V3_VERSION,
            len(self.active),
            cuda.MAX_STREAMS,
            0,
        ]))

    def close(self):
        try:
            self.client.stop()
        except (BrokenPipeError, OSError):
            pass
        self.client.connection.close()
        self.thread.join(timeout=2)
        self.server.close()
        if self.error is not None:
            raise self.error


class CudaRouteCaptureTests(unittest.TestCase):
    def test_capture_requires_explicit_hardware_confirmation(self):
        args = types.SimpleNamespace(execute=False, confirm=None)
        with self.assertRaisesRegex(cuda.CaptureError, "E_EXECUTE_GATE"):
            cuda.capture(args)

    def stage_plan(self):
        return {
            "expected_capabilities": cuda.CAPABILITIES,
            "expected_n_batch": cuda.N_BATCH,
            "expected_n_ctx_seq": cuda.N_CTX_SEQ,
            "expected_n_embd": cuda.N_EMBD,
            "expected_n_layer": cuda.N_LAYER,
            "expected_n_ubatch": cuda.N_UBATCH,
            "expected_max_streams": cuda.MAX_STREAMS,
        }

    def histories(self):
        return [[100 + index, 200 + index] for index in range(cuda.BATCH)]

    def test_stagev3_b8_geometry_stays_live_until_explicit_cleanup(self):
        stage = FakeStage()
        try:
            stage.client.hello(self.stage_plan())
            request_ids = list(range(1001, 1009))
            outputs, calls = cuda.run_generation(
                stage.client,
                self.histories(),
                request_ids,
                7,
            )
            self.assertEqual(stage.client.status()[0], 8)
            self.assertEqual(len(calls), 9)
            self.assertEqual(calls[0]["n_tokens"], 16)
            self.assertEqual([call["n_tokens"] for call in calls[1:]], [8] * 8)
            self.assertTrue(all(len(output) == 8 for output in outputs))
            cuda.remove_group(stage.client, request_ids, 7)
            self.assertEqual(stage.client.status()[0], 0)
        finally:
            stage.close()

    def test_cleanup_failure_is_fail_closed(self):
        stage = FakeStage(sticky_cleanup=True)
        try:
            stage.client.hello(self.stage_plan())
            request_ids = list(range(1001, 1009))
            cuda.run_generation(stage.client, self.histories(), request_ids, 7)
            with self.assertRaisesRegex(cuda.CaptureError, "cleanup.state_count"):
                cuda.remove_group(stage.client, request_ids, 7)
        finally:
            stage.close()

    def test_quality_prefill_is_position_major_and_eight_tokens(self):
        stage = FakeStage()
        try:
            stage.client.hello(self.stage_plan())
            histories = [
                list(range(10 + index, 14 + index + index % 3))
                for index in range(cuda.BATCH)
            ]
            outputs = cuda.run_quality_cohort(
                stage.client,
                histories,
                list(range(2001, 2009)),
                9,
            )
            self.assertTrue(all(len(output) == 8 for output in outputs))
            prefill = stage.calls[:-8]
            positions = [
                row[3]
                for call in prefill
                for row in call
            ]
            self.assertEqual(positions, sorted(positions))
            cuda.remove_group(stage.client, list(range(2001, 2009)), 9)
        finally:
            stage.close()

    def plan_commands(self):
        codec = ["/bin/echo", "codec"]
        worker = ["/bin/echo", "worker"]
        device = ["/bin/echo", "device"]
        process = ["/bin/echo", "process"]
        monolithic = ["/bin/echo", "monolithic"]
        return {
            "codec": {"argv": codec},
            "worker": {"argv": worker},
            "nvidia_smi": {
                "device_argv": device,
                "process_argv": process,
            },
            "mechanism_commands": {
                "desktop": [
                    codec,
                    worker,
                    device,
                    process,
                    device,
                    process,
                    device,
                    process,
                    monolithic,
                ],
                "op12": [["/bin/echo", "op12"]],
                "op15": [["/bin/echo", "op15"]],
            },
        }

    def test_full_mechanism_digest_binds_shared_nine_entry_matrix(self):
        plan = self.plan_commands()
        expected = cuda.sha256(cuda.canonical_bytes(plan["mechanism_commands"]))
        self.assertEqual(cuda.bind_mechanism_commands(plan, expected), expected)
        mutated = copy.deepcopy(plan)
        mutated["mechanism_commands"]["desktop"][3] = ["/bin/echo", "changed"]
        with self.assertRaisesRegex(cuda.CaptureError, "desktop.local"):
            cuda.bind_mechanism_commands(mutated, expected)

    def test_command_digest_mismatch_is_rejected(self):
        plan = self.plan_commands()
        with self.assertRaisesRegex(cuda.CaptureError, "mechanism_commands_sha256"):
            cuda.bind_mechanism_commands(plan, "0" * 64)

    def test_wrong_gpu_uuid_is_rejected(self):
        raw = (
            f"{cuda.CUDA_NAME}, GPU-00000000-0000-0000-0000-000000000000, "
            "16380, 1\n"
        ).encode("ascii")
        with self.assertRaisesRegex(cuda.CaptureError, "nvidia.device.uuid"):
            cuda.parse_device_probe(raw)

    def test_float_and_bool_do_not_pass_integer_gate(self):
        for value in (8.0, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(cuda.CaptureError, "E_INTEGER"):
                    cuda.integer(value, "test")

    def memory_cert(self):
        return {
            "compute_buffer_bytes": 100,
            "host_compute_buffer_bytes": 0,
            "host_context_buffer_bytes": 0,
            "host_model_buffer_bytes": 0,
            "kv_buffer_bytes": 200,
            "model_buffer_bytes": 700,
            "pid": 123,
            "role": "monov3",
            "schema": "layersplit-memory-breakdown-v1",
        }

    def placement(self):
        return {"compute_nodes": 20}

    def samples(self):
        return [
            {
                "host_swap_used_bytes": 10,
                "kind": "before",
                "nvml_process_used_bytes": 0,
                "sample_id": "a" * 64,
                "timestamp_ns": 10,
                "total_bytes": cuda.CUDA_MEMORY_TOTAL,
                "used_bytes": 1_000,
            },
            {
                "host_swap_used_bytes": 10,
                "kind": "ready",
                "nvml_process_used_bytes": 1_100,
                "sample_id": "b" * 64,
                "timestamp_ns": 20,
                "total_bytes": cuda.CUDA_MEMORY_TOTAL,
                "used_bytes": 2_000,
            },
            {
                "host_swap_used_bytes": 10,
                "kind": "after",
                "nvml_process_used_bytes": 0,
                "sample_id": "c" * 64,
                "timestamp_ns": 30,
                "total_bytes": cuda.CUDA_MEMORY_TOTAL,
                "used_bytes": 1_000,
            },
        ]

    def test_memory_split_is_derived_from_memory_cert(self):
        rows = cuda.make_memory_rows(
            self.samples(),
            self.memory_cert(),
            self.placement(),
            "d" * 64,
        )
        self.assertEqual(rows[1]["model_buffer_bytes"], 700)
        self.assertEqual(rows[1]["kv_buffer_bytes"], 200)
        self.assertEqual(rows[1]["process_used_bytes"], 900)
        self.assertEqual(rows[1]["state_count"], 8)
        phase_id = "cp0-r1-v23-a-only-test"
        normalized = cuda.normalized_memory_ready(rows[1], phase_id)
        self.assertEqual(normalized["acquisition_id"], phase_id)
        self.assertNotIn("process_pid", normalized)
        self.assertNotIn("sample_id", normalized)

    def test_forged_model_kv_split_exceeding_nvml_is_rejected(self):
        samples = self.samples()
        samples[1]["nvml_process_used_bytes"] = 999
        with self.assertRaisesRegex(cuda.CaptureError, "MEMORY_CERT_NVML"):
            cuda.make_memory_rows(
                samples,
                self.memory_cert(),
                self.placement(),
                "d" * 64,
            )

    def test_wrong_b8_and_continuation_count_are_rejected(self):
        histories = self.histories()
        continuations = [[1] * 8 for _ in range(8)]
        calls = [{
            "call_index": 0,
            "n_seqs": 8,
            "n_tokens": 8,
            "phase": "prefill",
        }] + [{
            "call_index": index,
            "n_seqs": 8,
            "n_tokens": 8,
            "phase": "decode",
        } for index in range(1, 9)]
        with self.assertRaisesRegex(cuda.CaptureError, "mechanics.prefill"):
            cuda.make_mechanics_rows(
                histories,
                continuations,
                calls,
                "d" * 64,
                100,
            )
        calls[0]["n_tokens"] = 16
        continuations[0].pop()
        with self.assertRaisesRegex(cuda.CaptureError, "length"):
            cuda.make_mechanics_rows(
                histories,
                continuations,
                calls,
                "d" * 64,
                100,
            )

    def test_stale_or_out_of_order_memory_events_are_rejected(self):
        with self.assertRaisesRegex(cuda.CaptureError, "E_SAMPLE_ORDER"):
            cuda.validate_samples(self.samples(), 11, 40)
        values = self.samples()
        values[1]["timestamp_ns"] = 30
        with self.assertRaisesRegex(cuda.CaptureError, "E_SAMPLE_ORDER"):
            cuda.validate_samples(values, 1, 40)

    def test_nonzero_swap_growth_is_rejected(self):
        values = self.samples()
        values[2]["host_swap_used_bytes"] += 1
        with self.assertRaisesRegex(cuda.CaptureError, "swap_growth"):
            cuda.validate_samples(values, 1, 40)

    def runtime_record(self):
        dependency = {
            "build_id": None,
            "ctime_ns": 1,
            "device_id": 2,
            "inode": 3,
            "mode": 0o100755,
            "mtime_ns": 4,
            "path": "/runtime/llama-layersplit",
            "size": 5,
        }
        return {
            "boot_id": "11111111-1111-4111-8111-111111111111",
            "launcher_path": "/runtime/llama-layersplit",
            "loaded_repo_component_ids": ["cuda_route.launcher"],
            "pid": 123,
            "schema": "s39-runtime-process-source-v1",
            "start_ticks": 99,
            "system_dependencies": [dependency],
        }, dependency

    def test_missing_runtime_identity_is_rejected(self):
        plan = {
            "worker": {
                "runtime_component_ids": ["cuda_route.launcher"],
                "runtime_executable": {"path": "/runtime/llama-layersplit"},
            },
        }
        component = {
            "path": "/runtime/llama-layersplit",
            "stat": {
                "ctime_ns": 1,
                "device_id": 2,
                "inode": 3,
                "mode": 0o100755,
                "mtime_ns": 4,
                "size": 5,
            },
        }
        with self.assertRaisesRegex(cuda.CaptureError, "runtime_process.count"):
            cuda.parse_runtime_process(
                b"",
                plan,
                "11111111-1111-4111-8111-111111111111",
                10,
                30,
                20,
                123,
                99,
                "a" * 64,
                {"cuda_route.launcher": component},
            )

    def test_runtime_identity_binds_boot_pid_path_components_and_stats(self):
        value, dependency = self.runtime_record()
        raw = cuda.RUNTIME_PREFIX + cuda.canonical_bytes(value)
        plan = {
            "worker": {
                "runtime_component_ids": ["cuda_route.launcher"],
                "runtime_executable": {"path": dependency["path"]},
            },
        }
        component = {
            "path": dependency["path"],
            "stat": {
                key: dependency[key]
                for key in cuda.STAT_KEYS
            },
        }
        parsed = cuda.parse_runtime_process(
            raw,
            plan,
            value["boot_id"],
            10,
            30,
            20,
            123,
            99,
            "a" * 64,
            {"cuda_route.launcher": component},
        )
        self.assertEqual(parsed["pid"], 123)
        self.assertEqual(parsed["identity_probe_sha256"], "a" * 64)
        mutated = raw.replace(value["boot_id"].encode(), b"bad-boot")
        with self.assertRaisesRegex(cuda.CaptureError, "boot_id"):
            cuda.parse_runtime_process(
                mutated,
                plan,
                value["boot_id"],
                10,
                30,
                20,
                123,
                99,
                "a" * 64,
                {"cuda_route.launcher": component},
            )

        value["start_ticks"] = 100
        with self.assertRaisesRegex(cuda.CaptureError, "start_ticks"):
            cuda.parse_runtime_process(
                cuda.RUNTIME_PREFIX + cuda.canonical_bytes(value),
                plan,
                value["boot_id"],
                10,
                30,
                20,
                123,
                99,
                "a" * 64,
                {"cuda_route.launcher": component},
            )

    def test_live_proc_start_ticks_are_independently_read(self):
        self.assertGreater(cuda.read_process_start_ticks(os.getpid()), 0)

    def placement_record(self):
        return {
            "compute_by_buffer_type": {"CUDA0": 2},
            "compute_by_op": {"MUL_MAT": 2},
            "compute_by_op_and_buffer": {"MUL_MAT": {"CUDA0": 2}},
            "compute_nodes": 2,
            "copy_by_buffer_type": {"CUDA0": 1},
            "copy_nodes": 1,
            "layer_end": cuda.N_LAYER,
            "layer_start": 0,
            "metadata_nodes": 3,
            "missing_buffer_compute_nodes": 0,
            "mode": "monov3",
            "n_layer": cuda.N_LAYER,
            "pid": 123,
            "role": "monov3",
            "run_rc": 0,
            "schema": "layersplit-scheduled-placement-v2",
            "status": "SCHEDULED_PLACEMENT_OK",
        }

    def test_placement_requires_full_cuda0_route_without_cpu_fallback(self):
        placement = self.placement_record()
        raw = cuda.PLACEMENT_PREFIX + cuda.canonical_bytes(placement)
        self.assertEqual(cuda.parse_placement(raw, 123)["compute_nodes"], 2)
        placement["compute_by_op_and_buffer"]["MUL_MAT"] = {"CPU": 2}
        raw = cuda.PLACEMENT_PREFIX + cuda.canonical_bytes(placement)
        with self.assertRaisesRegex(cuda.CaptureError, "backend"):
            cuda.parse_placement(raw, 123)

    def test_fake_nvidia_probe_persists_raw_and_checks_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plan = {
                "nvidia_smi": {
                    "device_argv": ["/probe", "device"],
                    "process_argv": ["/probe", "process"],
                    "timeout_ms": 1000,
                },
            }
            outputs = [
                (
                    f"{cuda.CUDA_NAME}, {cuda.CUDA_UUID}, 16380, 1000\n"
                ).encode("ascii"),
                b"123, 900\n",
            ]
            with (
                mock.patch.object(cuda, "run_command", side_effect=outputs),
                mock.patch.object(cuda, "read_swap", return_value=(0, b"swap\n")),
                mock.patch.object(cuda, "monotonic_ns", side_effect=[10, 20]),
            ):
                value = cuda.take_memory_sample(plan, "ready", 123, root)
            self.assertEqual(value["nvml_process_used_bytes"], 900 * 1024 * 1024)
            self.assertTrue((root / "ready.device.stdout").is_file())
            self.assertTrue((root / "ready.sample.json").is_file())


if __name__ == "__main__":
    unittest.main()
