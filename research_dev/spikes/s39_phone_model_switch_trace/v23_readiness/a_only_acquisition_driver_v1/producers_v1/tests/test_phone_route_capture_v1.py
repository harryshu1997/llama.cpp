#!/usr/bin/env python3

import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "phone_route_capture_v1.py"
SPEC = importlib.util.spec_from_file_location("phone_route_capture_v1", SOURCE)
phone = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phone)


def packed(fmt, values):
    values = tuple(values)
    return struct.pack("<" + fmt * len(values), *values)


def read_exact(connection, size):
    value = bytearray()
    while len(value) < size:
        block = connection.recv(size - len(value))
        if not block:
            raise EOFError
        value.extend(block)
    return bytes(value)


class FakeStage:
    def __init__(self):
        client, server = socket.socketpair()
        self.client = phone.StageClient(client)
        self.server = server
        self.error = None
        self.calls = []
        self.removed = []
        self.active = set()
        self.thread = threading.Thread(target=self._serve)
        self.thread.start()

    def _serve(self):
        try:
            while True:
                opcode = struct.unpack("<i", read_exact(self.server, 4))[0]
                if opcode == phone.STAGE_V3_HELLO:
                    self.server.sendall(packed("i", [
                        phone.STAGE_V3_MAGIC,
                        phone.STAGE_V3_VERSION,
                        0,
                        phone.N_LAYER,
                        phone.N_LAYER,
                        phone.N_EMBD,
                        phone.MAX_STREAMS,
                        phone.N_CTX_SEQ,
                        phone.N_BATCH,
                        phone.N_UBATCH,
                        phone.STAGE_V3_REQUIRED_CAPABILITIES,
                    ]))
                elif opcode == phone.STAGE_V3_IDENTITY:
                    self.server.sendall(
                        packed("i", [
                            phone.STAGE_IDENTITY_MAGIC,
                            phone.STAGE_IDENTITY_VERSION,
                            phone.FILE_TYPE,
                        ])
                        + bytes.fromhex(phone.MODEL_SHA256)
                    )
                elif opcode == phone.STAGE_V3_STATUS:
                    version = struct.unpack("<i", read_exact(self.server, 4))[0]
                    if version != phone.STAGE_V3_VERSION:
                        raise AssertionError(version)
                    self.server.sendall(packed("i", [
                        0,
                        phone.STAGE_V3_VERSION,
                        len(self.active),
                        phone.MAX_STREAMS,
                        0,
                    ]))
                elif opcode == phone.STAGE_V3_BATCH:
                    version, count, flags = struct.unpack(
                        "<3i",
                        read_exact(self.server, 12),
                    )
                    if version != phone.STAGE_V3_VERSION or flags != 0:
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
                elif opcode == phone.STAGE_V3_SEQ_REMOVE:
                    version, sequence = struct.unpack(
                        "<2i",
                        read_exact(self.server, 8),
                    )
                    request_id, epoch = struct.unpack(
                        "<2q",
                        read_exact(self.server, 16),
                    )
                    if version != phone.STAGE_V3_VERSION:
                        raise AssertionError(version)
                    self.active.discard(sequence)
                    self.removed.append((request_id, epoch, sequence))
                    self.server.sendall(packed("i", [
                        0,
                        phone.STAGE_V3_VERSION,
                        len(self.active),
                        phone.MAX_STREAMS,
                        0,
                    ]))
                elif opcode == phone.STAGE_STOP:
                    return
                else:
                    raise AssertionError(opcode)
        except BaseException as error:
            self.error = error

    def close(self):
        try:
            self.client.stop()
        finally:
            self.client.connection.close()
            self.thread.join(timeout=2)
            self.server.close()
        if self.error is not None:
            raise self.error


class PhoneRouteCaptureTests(unittest.TestCase):
    def test_probe_status_requires_idle_non_draining_route(self):
        for status, should_pass in (
            ((0, phone.MAX_STREAMS, False), True),
            ((1, phone.MAX_STREAMS, False), False),
            ((0, phone.MAX_STREAMS - 1, False), False),
            ((0, phone.MAX_STREAMS, True), False),
        ):
            client = mock.Mock()
            client.status.return_value = status
            with self.subTest(status=status):
                if should_pass:
                    phone.require_idle_route(client, "probe")
                else:
                    with self.assertRaises(phone.CaptureError):
                        phone.require_idle_route(client, "probe")
            client.status.assert_called_once_with()

    def test_probe_appends_live_worker_identity_to_static_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "probe.json"
            spec = {
                "before_argv": ["/probe", "--plan-json", "{}"],
                "cwd": str(root),
                "environment": {},
                "timeout_ms": 1000,
            }
            completed = subprocess.CompletedProcess(
                [],
                0,
                stdout=phone.canonical_bytes({"ok": True}),
                stderr=b"",
            )
            with mock.patch.object(
                phone.subprocess,
                "run",
                return_value=completed,
            ) as invoked:
                value, _ = phone.run_probe(
                    spec,
                    "before",
                    output,
                    "11111111-1111-4111-8111-111111111111",
                    123,
                    456,
                    789,
                    1011,
                )
            self.assertEqual(value, {"ok": True})
            self.assertEqual(
                invoked.call_args.args[0],
                [
                    "/probe",
                    "--plan-json",
                    "{}",
                    "--boot-id",
                    "11111111-1111-4111-8111-111111111111",
                    "--pid",
                    "123",
                    "--start-ticks",
                    "456",
                    "--network-pid",
                    "789",
                    "--network-start-ticks",
                    "1011",
                ],
            )
            self.assertEqual(output.read_bytes(), completed.stdout)

            bad = copy.deepcopy(spec)
            bad["before_argv"].extend(["--pid", "999"])
            with self.assertRaises(phone.CaptureError):
                phone.run_probe(
                    bad,
                    "before",
                    root / "bad.json",
                    "11111111-1111-4111-8111-111111111111",
                    123,
                    456,
                    789,
                    1011,
                )

    def test_fake_wire_mechanics_has_eight_decode_inputs_and_cleanup(self):
        wire = FakeStage()
        try:
            wire.client.hello({
                "expected_file_type": phone.FILE_TYPE,
                "expected_max_streams": phone.MAX_STREAMS,
                "expected_n_batch": phone.N_BATCH,
                "expected_n_ctx_seq": phone.N_CTX_SEQ,
                "expected_n_embd": phone.N_EMBD,
                "expected_n_ubatch": phone.N_UBATCH,
            })
            histories = [
                [100 + request, 200 + request]
                for request in range(phone.BATCH)
            ]
            request_ids = list(range(1001, 1009))
            continuations, calls, frames = phone.run_generation(
                wire.client,
                histories,
                request_ids,
                7,
                0,
            )
            self.assertEqual(len(calls), 9)
            self.assertEqual(
                [call["phase"] for call in calls],
                ["prefill"] + ["decode"] * 8,
            )
            self.assertEqual(
                sum(call["n_tokens"] for call in calls[1:]),
                64,
            )
            self.assertTrue(all(len(value) == 8 for value in continuations))
            self.assertEqual(len(frames), 9)
            phone.remove_group(wire.client, request_ids, 7)
            self.assertEqual(
                sorted(wire.removed),
                [(request_id, 7, sequence)
                 for sequence, request_id in enumerate(request_ids)],
            )
        finally:
            wire.close()

    def test_fake_wire_quality_is_position_major_and_exact_eight_tokens(self):
        wire = FakeStage()
        try:
            wire.client.hello({
                "expected_file_type": phone.FILE_TYPE,
                "expected_max_streams": phone.MAX_STREAMS,
                "expected_n_batch": phone.N_BATCH,
                "expected_n_ctx_seq": phone.N_CTX_SEQ,
                "expected_n_embd": phone.N_EMBD,
                "expected_n_ubatch": phone.N_UBATCH,
            })
            histories = [
                list(range(100 + request, 103 + request + (request % 3)))
                for request in range(phone.BATCH)
            ]
            request_ids = list(range(2001, 2009))
            outputs, frames = phone.run_quality_cohort(
                wire.client,
                histories,
                request_ids,
                9,
                0,
            )
            self.assertTrue(all(len(value) == 8 for value in outputs))
            self.assertTrue(all(len(call) <= phone.N_UBATCH for call in wire.calls))
            prefill = wire.calls[:-8]
            positions = [
                position
                for call in prefill
                for _, _, _, position, _ in call
            ]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(len(frames), len(wire.calls))
            phone.remove_group(wire.client, request_ids, 9)
        finally:
            wire.close()

    def mechanics(self):
        histories = [[index, index + 1] for index in range(phone.BATCH)]
        continuations = [
            list(range(index, index + 8))
            for index in range(phone.BATCH)
        ]
        calls = [
            {
                "call_index": 0,
                "n_seqs": 8,
                "n_tokens": 16,
                "phase": "prefill",
            }
        ] + [
            {
                "call_index": index + 1,
                "n_seqs": 8,
                "n_tokens": 8,
                "phase": "decode",
            }
            for index in range(8)
        ]
        return phone.make_mechanics_rows(
            histories,
            continuations,
            calls,
            "a" * 64,
            100,
        )

    def corpus(self):
        return [
            {
                "choices": ["a", "b", "c", "d"],
                "dataset": "cais/mmlu",
                "dataset_revision": "b" * 40,
                "expected_answer": index % 4,
                "item_index": index,
                "question": f"question {index}",
                "source_row": index,
                "subject": "subject",
            }
            for index in range(64)
        ]

    def test_bridge_digest_binds_normalized_acquisition_row(self):
        phase_id = "cp0-r1-v23-a-only-test"
        mechanics = self.mechanics()
        rows = phone.make_bridge_rows(mechanics, 500, phase_id)
        request = mechanics[1]
        normalized = {
            "acquisition_id": phase_id,
            **{key: value for key, value in request.items() if key != "event_ns"},
            "role": f"model.{phone.MODEL_ID}.mechanics.phone",
        }
        self.assertEqual(
            rows[0]["phone_request_sha256"],
            phone.sha256(phone.canonical_bytes(normalized)),
        )
        mutated = copy.deepcopy(request)
        mutated["continuation_tokens"][0] += 1
        self.assertNotEqual(
            rows[0]["phone_request_sha256"],
            phone.sha256(phone.canonical_bytes({
                "acquisition_id": phase_id,
                **{
                    key: value
                    for key, value in mutated.items()
                    if key != "event_ns"
                },
                "role": f"model.{phone.MODEL_ID}.mechanics.phone",
            })),
        )

    def test_quality_digest_binds_corpus_wrapper_and_prompt(self):
        phase_id = "cp0-r1-v23-a-only-test"
        corpus = self.corpus()
        rows = phone.make_quality_rows(
            corpus,
            ["A"] * 64,
            "c" * 64,
            1000,
            phase_id,
        )
        normalized = {
            "acquisition_id": phase_id,
            "kind": "item",
            "role": "quality.corpus",
            **corpus[0],
        }
        self.assertEqual(
            rows[0]["corpus_item_sha256"],
            phone.sha256(phone.canonical_bytes(normalized)),
        )
        self.assertEqual(
            rows[0]["prompt_sha256"],
            phone.sha256(phone.prompt_for(corpus[0]).encode("utf-8")),
        )

    def test_codec_preserves_one_response_per_request(self):
        requests = [
            {
                "op": "tokenize",
                "request_id": request_id,
                "schema": "layersplit-token-codec-request-v1",
                "text": f"prompt {request_id}",
            }
            for request_id in (1, 2)
        ]
        stdout = b"".join(
            phone.canonical_bytes({
                "model_sha256": phone.MODEL_SHA256,
                "op": "tokenize",
                "request_id": request_id,
                "schema": "layersplit-token-codec-response-v1",
                "tokens": [request_id],
            })
            for request_id in (1, 2)
        )
        completed = subprocess.CompletedProcess(
            ["codec"],
            0,
            stdout,
            b"",
        )
        codec = {
            "argv": ["codec"],
            "cwd": "/tmp",
            "environment": {},
            "timeout_ms": 1000,
        }
        with mock.patch.object(phone.subprocess, "run", return_value=completed):
            responses = phone.invoke_codec(codec, requests)
        self.assertEqual(
            [response["request_id"] for response in responses],
            [1, 2],
        )

    def worker_log(self, name, total_rows=72):
        op15 = name == "op15"
        layer_start, layer_end = (
            phone.OP15_EXECUTED if op15 else phone.OP12_EXECUTED
        )
        op_map = {"MUL_MAT": {"OpenCL": 20}}
        if op15:
            op_map["GET_ROWS"] = {"CPU": 2}
        session = {
            "compute_by_op_and_buffer": op_map,
            "device_boot_id": name + "-boot",
            "expected_backend": "GPUOpenCL",
            "layer_end": layer_end,
            "layer_start": layer_start,
            "missing_buffer_compute_nodes": 0,
            "n_layer": phone.N_LAYER,
            "placement_status": "SCHEDULED_PLACEMENT_OK",
            "proto_version": 2,
            "reset_applied": False,
            "schema": "ls-stagenet-session-v2",
            "session_end": "STOP",
            "session_id": 1,
            "steps_session": total_rows,
            "steps_total": total_rows,
            "worker_boot_nonce": "0123456789abcdef",
            "worker_pid": 123,
        }
        placement = {
            "compute_by_buffer_type": {
                "CPU": 2,
                "OpenCL": 20,
            } if op15 else {"OpenCL": 20},
            "compute_by_op": {
                key: sum(value.values())
                for key, value in op_map.items()
            },
            "compute_by_op_and_buffer": op_map,
            "compute_nodes": 22 if op15 else 20,
            "copy_by_buffer_type": {},
            "copy_nodes": 0,
            "layer_end": layer_end,
            "layer_start": layer_start,
            "metadata_nodes": 10,
            "missing_buffer_compute_nodes": 0,
            "mode": "stagenet" if op15 else "tailv3",
            "n_layer": phone.N_LAYER,
            "pid": 123,
            "role": "phone_stage" if op15 else "host_tail_v3",
            "run_rc": 0,
            "schema": "layersplit-scheduled-placement-v2",
            "status": "SCHEDULED_PLACEMENT_OK",
        }
        raw = (
            b"SESSIONCERT " + phone.canonical_bytes(session)
            + b"PLACEMENTCERT " + phone.canonical_bytes(placement)
        )
        expected = {
            "boot_id": name + "-boot",
            "executed_layers": [layer_start, layer_end],
        }
        return raw, expected

    def test_worker_certs_validate_counts_but_emit_unique_nodes(self):
        raw, expected = self.worker_log("op15")
        _, placement, nodes = phone.parse_worker_log(
            raw,
            "op15",
            expected,
            72,
        )
        self.assertEqual(placement["compute_nodes"], 22)
        self.assertEqual(nodes, [("GET_ROWS", "CPU"), ("MUL_MAT", "OpenCL")])

    def test_worker_cert_missing_partial_wrong_range_and_cpu_fail_closed(self):
        raw, expected = self.worker_log("op12")
        cases = [
            raw.split(b"PLACEMENTCERT ", 1)[0],
            raw.rstrip(b"\n"),
            raw.replace(b'"layer_start":30', b'"layer_start":29', 1),
            raw.replace(
                b'"MUL_MAT":{"OpenCL":20}',
                b'"MUL_MAT":{"CPU":20}',
            ),
        ]
        for candidate in cases:
            with self.subTest(candidate=candidate[-80:]):
                with self.assertRaises(phone.CaptureError):
                    phone.parse_worker_log(
                        candidate,
                        "op12",
                        expected,
                        72,
                    )

    def direct_log(self, expected_frames):
        records = []
        total_rows = 0
        total_bytes = 0
        for expected in expected_frames:
            frame = {
                **expected,
                "activation_payload_bytes": expected["rows"] * phone.N_EMBD * 4,
                "payload_sha256": "d" * 64,
                "schema": "ls-stage-direct-frame-v1",
            }
            total_rows += expected["rows"]
            total_bytes += frame["activation_payload_bytes"]
            records.append(b"DIRECTFRAME " + phone.canonical_bytes(frame))
        cert = {
            "activation_payload_bytes": total_bytes,
            "batches": len(expected_frames),
            "cut_layer": 30,
            "file_type": phone.FILE_TYPE,
            "head_endpoint": "127.0.0.1:1001",
            "host_activation_payload_bytes": 0,
            "layer_end": phone.N_LAYER,
            "layer_start": 0,
            "model_sha256": phone.MODEL_SHA256,
            "n_embd": phone.N_EMBD,
            "n_layer": phone.N_LAYER,
            "rows": total_rows,
            "run_rc": 0,
            "schema": "ls-stage-direct-relay-v1",
            "status": "DIRECT_RELAY_OK",
            "tail_endpoint": "192.0.2.2:1002",
        }
        records.append(b"DIRECTCERT " + phone.canonical_bytes(cert))
        return b"".join(records)

    def test_direct_frame_evidence_is_complete(self):
        expected = [
            {
                "call_index": 0,
                "hidden_width": phone.N_EMBD,
                "positions": [0, 0],
                "request_ids": [1, 2],
                "route_epochs": [4, 4],
                "rows": 2,
                "seq_ids": [0, 1],
            },
            {
                "call_index": 1,
                "hidden_width": phone.N_EMBD,
                "positions": [1, 1],
                "request_ids": [1, 2],
                "route_epochs": [4, 4],
                "rows": 2,
                "seq_ids": [0, 1],
            },
        ]
        raw = self.direct_log(expected)
        frames, cert = phone.parse_direct_frames(raw, expected)
        self.assertEqual(len(frames), 2)
        self.assertEqual(cert["rows"], 4)
        mutations = [
            raw.split(b"DIRECTFRAME ", 1)[0]
            + b"DIRECTFRAME "
            + raw.split(b"DIRECTFRAME ", 2)[2],
            raw.rstrip(b"\n"),
            raw.replace(b'"cut_layer":30', b'"cut_layer":29'),
        ]
        for candidate in mutations:
            with self.subTest(candidate=candidate[-80:]):
                with self.assertRaises(phone.CaptureError):
                    phone.parse_direct_frames(candidate, expected)

    def runtime_record(self, name, endpoint, boot, path, observed=150):
        return {
            "boot_id": boot,
            "bundle_id": name,
            "endpoint": endpoint,
            "launcher_path": path,
            "loaded_repo_component_ids": ["llama-layersplit"],
            "observed_ns": observed,
            "pid": 123,
            "start_ticks": 456,
            "system_dependencies": [{
                "build_id": None,
                "ctime_ns": 1,
                "device_id": 2,
                "inode": 3,
                "mode": stat.S_IFREG | 0o755,
                "mtime_ns": 4,
                "path": "/vendor/lib64/libc.so",
                "size": 5,
            }],
        }

    def test_runtime_process_is_raw_bound_and_interval_checked(self):
        path = "/data/local/tmp/llama-layersplit"
        value = self.runtime_record(
            "op12_stagenet",
            "op12",
            "boot",
            path,
        )
        raw = b"RUNTIMEPROCESS " + phone.canonical_bytes(value)
        spec = {
            "runtime_component_ids": ["llama-layersplit"],
            "runtime_executable_path": path,
        }
        parsed = phone.parse_runtime_process(
            raw,
            "op12_stagenet",
            "op12",
            "boot",
            spec,
            100,
            200,
            "a" * 64,
            123,
            456,
        )
        self.assertEqual(parsed, {**value, "identity_probe_sha256": "a" * 64})
        for key, bad in (
            ("launcher_path", "/wrong"),
            ("boot_id", "old-boot"),
            ("observed_ns", 99),
        ):
            mutated = copy.deepcopy(value)
            mutated[key] = bad
            with self.subTest(key=key):
                with self.assertRaises(phone.CaptureError):
                    phone.parse_runtime_process(
                        b"RUNTIMEPROCESS " + phone.canonical_bytes(mutated),
                        "op12_stagenet",
                        "op12",
                        "boot",
                        spec,
                        100,
                        200,
                        "a" * 64,
                        123,
                        456,
                    )
        for key, bad in (("pid", 124), ("start_ticks", 457)):
            mutated = copy.deepcopy(value)
            mutated[key] = bad
            with self.subTest(key=key):
                with self.assertRaises(phone.CaptureError):
                    phone.parse_runtime_process(
                        b"RUNTIMEPROCESS " + phone.canonical_bytes(mutated),
                        "op15_direct_relay",
                        "op15",
                        "boot",
                        spec,
                        100,
                        200,
                        "b" * 64,
                        123,
                        456,
                    )

    def test_runtime_interface_delta_starts_at_phase_local_probe(self):
        expected = {
            "boot_id": "boot",
            "expected_worker_executable_path": "/data/local/tmp/worker",
            "loaded_shard_path": "/data/local/tmp/weights.gguf",
            "serial": "serial",
        }
        peer = {
            "interface": "wlan0",
            "local_ipv4": "192.0.2.1",
            "peer_ipv4": "192.0.2.2",
            "socket_peer_observed": True,
        }
        before = {
            "active_sequences": 0,
            "available_bytes": 1_000_000_000,
            "direct_peer": peer,
            "gpu_max_millic": 50_000,
            "interface": {
                "ipv4": "192.0.2.1",
                "name": "wlan0",
                "rx_bytes": 200,
                "tx_bytes": 300,
            },
            "process_swap_bytes": 0,
            "worker_pid": 123,
            "worker_start_ticks": 456,
        }
        after = copy.deepcopy(before)
        after["interface"]["rx_bytes"] = 1200
        after["interface"]["tx_bytes"] = 2300
        session = {
            "proto_version": 2,
            "worker_boot_nonce": "0123456789abcdef",
            "worker_pid": 123,
        }
        baseline = {
            "interface": "wlan0",
            "rx_bytes": 100,
            "tx_bytes": 150,
        }
        result = phone.runtime_record(
            expected,
            before,
            after,
            session,
            baseline,
            7,
        )
        self.assertEqual(
            result["interface_before"],
            {
                "interface": "wlan0",
                "rx_bytes": 200,
                "tx_bytes": 300,
            },
        )
        mutated = copy.deepcopy(before)
        mutated["interface"]["rx_bytes"] = 99
        with self.assertRaises(phone.CaptureError):
            phone.runtime_record(
                expected,
                mutated,
                after,
                session,
                baseline,
                7,
            )

    def test_relay_process_probe_binds_remote_pid_ticks_argv_and_port(self):
        spec = {
            "expected_argv": [
                "/data/local/tmp/relay",
                "--listen",
                "12345",
            ],
            "expected_executable_path": "/data/local/tmp/relay",
            "expected_port": 12345,
        }
        row = {
            "adb_port": phone.PHONE_ADB_PORT,
            "adb_selector": "172.20.173.218:5555",
            "argv": spec["expected_argv"],
            "boot_id": "boot",
            "executable_path": spec["expected_executable_path"],
            "pid": 123,
            "port": spec["expected_port"],
            "schema": phone.RELAY_PROCESS_PROBE_SCHEMA,
            "start_ticks": 456,
        }
        phone.validate_relay_process_probe_row(
            row,
            spec,
            "boot",
            "172.20.173.218:5555",
        )
        for key, replacement in (
            ("adb_port", 5037),
            ("adb_selector", "172.20.59.72:5555"),
            ("pid", 0),
            ("start_ticks", 0),
            ("argv", ["/wrong"]),
            ("port", 12346),
        ):
            mutated = copy.deepcopy(row)
            mutated[key] = replacement
            with self.subTest(key=key):
                with self.assertRaises(phone.CaptureError):
                    phone.validate_relay_process_probe_row(
                        mutated,
                        spec,
                        "boot",
                        "172.20.173.218:5555",
                    )

    def test_probe_row_binds_network_owner_separately_from_worker(self):
        expected = {
            "boot_id": "11111111-1111-4111-8111-111111111111",
            "device": "device",
            "direct_peer_ipv4": "192.0.2.2",
            "expected_worker_executable_path": "/worker",
            "expected_worker_executable_sha256": "a" * 64,
            "interface": "wlan0",
            "loaded_shard_path": phone.PHONE_SHARD_PATH,
            "loaded_shard_sha256": phone.OP15_SHARD_SHA256,
            "local_ipv4": "192.0.2.1",
            "model": "model",
            "product": "product",
            "serial": "serial",
        }
        network = {
            "executable_path": "/relay",
            "executable_sha256": "b" * 64,
            "pid": 789,
            "role": "direct_relay",
            "start_ticks": 1011,
        }
        row = {
            "active_sequences": 0,
            "available_bytes": 1,
            "boot_id": expected["boot_id"],
            "device": expected["device"],
            "direct_peer": {
                "interface": "wlan0",
                "local_ipv4": "192.0.2.1",
                "peer_ipv4": "192.0.2.2",
                "socket_peer_observed": True,
            },
            "gpu_max_millic": 1,
            "interface": {
                "ipv4": "192.0.2.1",
                "name": "wlan0",
                "rx_bytes": 1,
                "tx_bytes": 1,
            },
            "loaded_shard_path": phone.PHONE_SHARD_PATH,
            "loaded_shard_sha256": phone.OP15_SHARD_SHA256,
            "model": "model",
            "model_id": phone.MODEL_ID,
            "model_sha256": phone.MODEL_SHA256,
            "network_executable_path": "/relay",
            "network_executable_sha256": "b" * 64,
            "network_pid": 789,
            "network_process_role": "direct_relay",
            "network_start_ticks": 1011,
            "process_swap_bytes": 0,
            "product": "product",
            "schema": phone.PROBE_SCHEMA,
            "serial": "serial",
            "system_swap_used_bytes": 100,
            "worker_executable_path": "/worker",
            "worker_executable_sha256": "a" * 64,
            "worker_pid": 123,
            "worker_start_ticks": 456,
        }
        phone.validate_probe_row(row, "op15", expected, network, "probe")
        for key, bad in (
            ("network_pid", 123),
            ("network_process_role", "stagenet_worker"),
            ("network_executable_path", "/worker"),
        ):
            mutated = copy.deepcopy(row)
            mutated[key] = bad
            with self.subTest(key=key):
                with self.assertRaises(phone.CaptureError):
                    phone.validate_probe_row(
                        mutated,
                        "op15",
                        expected,
                        network,
                        "probe",
                    )

    def write_executable(self, root):
        path = root / "launcher"
        path.write_bytes(b"#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        raw = path.read_bytes()
        return path, len(raw), phone.sha256(raw)

    def launch_plan(self, root):
        executable, size, checksum = self.write_executable(root)
        history = root / "history.json"
        history.write_bytes(phone.canonical_bytes({
            "histories": [[1] for _ in range(phone.BATCH)],
            "history_width": 1,
            "model_id": phone.MODEL_ID,
            "model_sha256": phone.MODEL_SHA256,
            "request_ids": list(range(phone.BATCH)),
            "route_epoch": 1,
            "schema": phone.HISTORY_SCHEMA,
        }))
        def managed_command(bundle, endpoint, route):
            runtime_path = "/data/local/tmp/llama-layersplit"
            managed = {
                "android": {
                    "boot_id_source": "phase_fresh_snapshot",
                },
                "bundle_id": bundle,
                "components": [{
                    "component_id": "llama-layersplit",
                    "path": runtime_path,
                    "sha256": "e" * 64,
                }],
                "endpoint": endpoint,
                "launcher_component_id": "llama-layersplit",
                "mode": "android",
                "route": route,
                "schema": "s39-managed-runtime-launch-plan-v1",
            }
            inline = json.dumps(
                managed,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            return {
                "argv": [
                    str(executable),
                    "--plan-json",
                    inline,
                    "--plan-sha256",
                    phone.sha256(inline.encode("ascii")),
                ],
                "cwd": str(root),
                "environment": {},
                "launcher_bytes": size,
                "launcher_sha256": checksum,
                "runtime_component_ids": ["llama-layersplit"],
                "runtime_executable_path": runtime_path,
                "runtime_executable_sha256": "e" * 64,
                "shutdown_timeout_ms": 1000,
                "startup_timeout_ms": 1000,
            }
        relay_runtime_argv = [
            "/data/local/tmp/llama-layersplit",
            "--listen",
            "12345",
            "--head",
            "127.0.0.1:1001",
            "--tail",
            "192.0.2.2:1002",
            "--emit-direct-frames",
        ]
        relay_process_probe = {
            "argv": [
                str(executable),
                "--adb",
                str(executable),
                "--adb-port",
                str(phone.PHONE_ADB_PORT),
                "--adb-selector",
                "172.20.173.218:5555",
                "--adb-sha256",
                checksum,
                "--expected-executable",
                relay_runtime_argv[0],
                "--expected-argv-json",
                json.dumps(
                    relay_runtime_argv,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                "--expected-port",
                "12345",
            ],
            "cwd": str(root),
            "environment": {},
            "expected_argv": relay_runtime_argv,
            "expected_executable_path": relay_runtime_argv[0],
            "expected_port": 12345,
            "launcher_bytes": size,
            "launcher_sha256": checksum,
            "timeout_ms": 1000,
        }
        phones = {}
        for name, stored, executed, shard, local, peer in (
            (
                "op15",
                phone.OP15_STORED,
                phone.OP15_EXECUTED,
                phone.OP15_SHARD_SHA256,
                "192.0.2.1",
                "192.0.2.2",
            ),
            (
                "op12",
                phone.OP12_STORED,
                phone.OP12_EXECUTED,
                phone.OP12_SHARD_SHA256,
                "192.0.2.2",
                "192.0.2.1",
            ),
        ):
            phones[name] = {
                "adb_selector": {
                    "op15": "172.20.173.218:5555",
                    "op12": "172.20.59.72:5555",
                }[name],
                "boot_id_source": "phase_fresh_snapshot",
                "device": name + "-device",
                "direct_peer_ipv4": peer,
                "executed_layers": executed,
                "expected_worker_executable_path": (
                    "/data/local/tmp/llama-layersplit"
                ),
                "expected_worker_executable_sha256": "e" * 64,
                "interface": "wlan0",
                "loaded_shard_path": phone.PHONE_SHARD_PATH,
                "loaded_shard_sha256": shard,
                "local_ipv4": local,
                "model": name + "-model",
                "product": name + "-product",
                "serial": name + "-serial",
                "stored_layers": stored,
            }
        processes = {
            "op12_stagenet": managed_command(
                "op12_stagenet",
                "op12",
                {
                    "kind": "stagenet_worker",
                    "layer_end": 40,
                    "layer_start": 30,
                    "model_path": phone.PHONE_SHARD_PATH,
                    "model_sha256": phone.MODEL_SHA256,
                },
            ),
            "op15_direct_relay": managed_command(
                "op15_direct_relay",
                "op15",
                {
                    "emit_direct_frames": True,
                    "kind": "direct_relay",
                    "listen_port": 12345,
                },
            ),
            "op15_stagenet": managed_command(
                "op15_stagenet",
                "op15",
                {
                    "kind": "stagenet_worker",
                    "layer_end": 30,
                    "layer_start": 0,
                    "model_path": phone.PHONE_SHARD_PATH,
                    "model_sha256": phone.MODEL_SHA256,
                },
            ),
        }
        probes = {}
        for name in ("op12", "op15"):
            expected = phones[name]
            worker_path = expected["expected_worker_executable_path"]
            network_path = (
                relay_runtime_argv[0] if name == "op15" else worker_path
            )
            network_sha256 = (
                processes["op15_direct_relay"]["runtime_executable_sha256"]
                if name == "op15"
                else expected["expected_worker_executable_sha256"]
            )
            process_argv = [worker_path]
            probe_plan = {
                "android": {
                    "adb_path": str(executable),
                    "adb_port": phone.PHONE_ADB_PORT,
                    "adb_selector": expected["adb_selector"],
                    "adb_sha256": checksum,
                    "boot_id_source": "phase_fresh_snapshot",
                    "device": expected["device"],
                    "model": expected["model"],
                    "physical_serial": expected["serial"],
                    "product": expected["product"],
                },
                "capture_schema": phone.PROBE_SCHEMA,
                "model_id": phone.MODEL_ID,
                "model_sha256": phone.MODEL_SHA256,
                "network_process": {
                    "argv": (
                        relay_runtime_argv if name == "op15" else process_argv
                    ),
                    "artifact": {
                        "path": network_path,
                        "sha256": network_sha256,
                    },
                    "executable_path": network_path,
                    "role": (
                        "direct_relay"
                        if name == "op15"
                        else "stagenet_worker"
                    ),
                },
                "process": {
                    "argv": process_argv,
                    "executable_path": worker_path,
                },
                "schema": "s39-phone-runtime-probe-plan-v1",
                "shard_artifact": {
                    "path": expected["loaded_shard_path"],
                    "sha256": expected["loaded_shard_sha256"],
                },
                "stage_v3": {
                    "expected_active_sequences": 0,
                    "source": "relay_owned_status",
                },
                "telemetry": {
                    "direct_peer_ipv4": expected["direct_peer_ipv4"],
                    "interface": expected["interface"],
                    "local_ipv4": expected["local_ipv4"],
                },
                "worker_artifact": {
                    "path": worker_path,
                    "sha256": expected["expected_worker_executable_sha256"],
                },
            }
            inline = json.dumps(
                probe_plan,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            argv = [
                str(executable),
                "--plan-json",
                inline,
                "--plan-sha256",
                phone.sha256(inline.encode("ascii")),
                "--capture-compatible",
            ]
            probes[name] = {
                "after_argv": list(argv),
                "before_argv": list(argv),
                "cwd": str(root),
                "environment": {},
                "launcher_bytes": size,
                "launcher_sha256": checksum,
                "timeout_ms": 1000,
            }
        return {
            "codec": {
                "argv": [
                    str(executable),
                    "--model",
                    "/model.gguf",
                    "--model-sha256",
                    phone.MODEL_SHA256,
                ],
                "cwd": str(root),
                "environment": {},
                "executable_bytes": size,
                "executable_sha256": checksum,
                "timeout_ms": 1000,
            },
            "expected_file_type": phone.FILE_TYPE,
            "expected_max_streams": phone.MAX_STREAMS,
            "expected_n_batch": phone.N_BATCH,
            "expected_n_ctx_seq": phone.N_CTX_SEQ,
            "expected_n_embd": phone.N_EMBD,
            "expected_n_layer": phone.N_LAYER,
            "expected_n_ubatch": phone.N_UBATCH,
            "history_path": str(history),
            "history_sha256": phone.sha256(history.read_bytes()),
            "mechanism_commands": {
                "desktop": [
                    [
                        str(executable),
                        "--model",
                        "/model.gguf",
                        "--model-sha256",
                        phone.MODEL_SHA256,
                    ],
                    ["/cuda-route"],
                    ["/nvidia-device", "before"],
                    ["/nvidia-process", "before"],
                    ["/nvidia-device", "ready"],
                    ["/nvidia-process", "ready"],
                    ["/nvidia-device", "after"],
                    ["/nvidia-process", "after"],
                    ["/cuda-monolithic"],
                ],
                "op12": [
                    list(processes["op12_stagenet"]["argv"]),
                    list(probes["op12"]["before_argv"]),
                    list(probes["op12"]["after_argv"]),
                ],
                "op15": [
                    list(processes["op15_stagenet"]["argv"]),
                    list(processes["op15_direct_relay"]["argv"]),
                    list(relay_process_probe["argv"]),
                    list(probes["op15"]["before_argv"]),
                    list(probes["op15"]["after_argv"]),
                ],
            },
            "model_id": phone.MODEL_ID,
            "model_sha256": phone.MODEL_SHA256,
            "phones": phones,
            "probes": probes,
            "processes": processes,
            "quality_corpus_content_sha256": "f" * 64,
            "relay_process_probe": relay_process_probe,
            "relay_host": "127.0.0.1",
            "relay_port": 12345,
            "route_epoch": 1,
            "schema": phone.PLAN_SCHEMA,
        }

    def test_plan_binds_launcher_shards_ranges_and_peers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.launch_plan(root)
            path = root / "plan.json"
            path.write_bytes(phone.canonical_bytes(plan))
            phone.load_plan(path)
            mutations = [
                ("launcher", ("processes", "op15_stagenet", "launcher_sha256"), "0" * 64),
                ("shard", ("phones", "op12", "loaded_shard_sha256"), "0" * 64),
                ("range", ("phones", "op15", "executed_layers"), [0, 29]),
                ("peer", ("phones", "op15", "direct_peer_ipv4"), "192.0.2.9"),
                (
                    "worker-runtime",
                    (
                        "processes",
                        "op12_stagenet",
                        "runtime_executable_sha256",
                    ),
                    "0" * 64,
                ),
                (
                    "direct-frame-flag",
                    ("processes", "op15_direct_relay", "argv"),
                    [str(root / "launcher")],
                ),
            ]
            for name, keys, replacement in mutations:
                mutated = copy.deepcopy(plan)
                target = mutated
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = replacement
                candidate = root / f"{name}.json"
                candidate.write_bytes(phone.canonical_bytes(mutated))
                with self.subTest(name=name):
                    with self.assertRaises(phone.CaptureError):
                        phone.load_plan(candidate)

    def test_process_boot_ids_are_bound_only_from_fresh_runtime_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self.launch_plan(Path(temporary))
            runtime = {
                "op12": {"boot_id": "fresh-op12"},
                "op15": {"boot_id": "fresh-op15"},
            }
            specs = phone.bind_process_boot_ids(plan, runtime)
            self.assertEqual(
                specs["op12_stagenet"]["argv"][-2:],
                ["--boot-id", "fresh-op12"],
            )
            self.assertEqual(
                specs["op15_direct_relay"]["argv"][-2:],
                ["--boot-id", "fresh-op15"],
            )
            self.assertNotIn(
                "fresh-op12",
                plan["processes"]["op12_stagenet"]["argv"],
            )
            mutated = copy.deepcopy(plan)
            mutated["processes"]["op12_stagenet"]["argv"].extend(
                ["--boot-id", "stale"]
            )
            with self.assertRaises(phone.CaptureError):
                phone.bind_process_boot_ids(mutated, runtime)

    def test_histories_argument_is_exact_bound_to_launch_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.launch_plan(root)
            history_path = Path(plan["history_path"])
            histories, route_epoch, raw = phone.load_bound_histories(
                str(history_path),
                plan,
            )
            self.assertEqual(len(histories), phone.BATCH)
            self.assertEqual(route_epoch, 1)
            self.assertEqual(phone.sha256(raw), plan["history_sha256"])
            alternate = root / "alternate-history.json"
            alternate.write_bytes(raw)
            with self.assertRaisesRegex(
                phone.CaptureError,
                "histories.launch_path",
            ):
                phone.load_bound_histories(str(alternate), plan)
            history_path.write_bytes(b"{}\n")
            with self.assertRaises(phone.CaptureError):
                phone.load_bound_histories(str(history_path), plan)

    def test_mechanism_digest_is_derived_from_executed_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self.launch_plan(Path(temporary))
            mechanisms = plan["mechanism_commands"]
            checksum = phone.sha256(phone.canonical_bytes(mechanisms))
            self.assertEqual(
                phone.bind_mechanism_commands(plan, checksum),
                checksum,
            )
            changed = copy.deepcopy(mechanisms)
            changed["desktop"][0].append("--changed")
            with self.assertRaises(phone.CaptureError):
                phone.bind_mechanism_commands(
                    plan,
                    phone.sha256(phone.canonical_bytes(changed)),
                )
            plan["mechanism_commands"]["op15"][0].append("--changed")
            with self.assertRaises(phone.CaptureError):
                phone.bind_mechanism_commands(
                    plan,
                    phone.sha256(
                        phone.canonical_bytes(plan["mechanism_commands"])
                    ),
                )

    def test_phase_lock_must_exist_and_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase_id = "cp0-r1-v23-a-only-test"
            with self.assertRaises(phone.CaptureError):
                phone.load_phase(root, phase_id)
            (root / "phase_lock.jsonl").write_bytes(phone.canonical_bytes({
                "phase": phone.PHASE,
                "phase_id": "wrong",
            }))
            with self.assertRaises(phone.CaptureError):
                phone.load_phase(root, phase_id)

    def test_wait_log_marker_detects_ready_and_missing_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "ready"
            ready.write_text(
                "#!/usr/bin/env python3\n"
                "import time\n"
                "print('[stagenet] listening on 0.0.0.0:1', flush=True)\n"
                "time.sleep(10)\n",
                encoding="ascii",
            )
            ready.chmod(0o755)
            spec = {
                "argv": [str(ready)],
                "cwd": str(root),
                "environment": dict(os.environ),
                "shutdown_timeout_ms": 100,
                "startup_timeout_ms": 1000,
            }
            managed = phone.ManagedProcess("ready", spec, root)
            managed.start()
            try:
                phone.wait_log_marker(managed, b"[stagenet] listening on ")
            finally:
                managed.kill()

            silent = root / "silent"
            silent.write_text(
                "#!/usr/bin/env python3\n"
                "import time\n"
                "time.sleep(10)\n",
                encoding="ascii",
            )
            silent.chmod(0o755)
            spec["argv"] = [str(silent)]
            spec["startup_timeout_ms"] = 50
            managed = phone.ManagedProcess("silent", spec, root)
            managed.start()
            try:
                with self.assertRaises(phone.CaptureError):
                    phone.wait_log_marker(managed, b"ready")
            finally:
                managed.kill()

    def test_cli_refuses_without_explicit_hardware_confirmation(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                os.fspath(SOURCE),
                "--output",
                "/tmp/unused-output",
                "--phase-id",
                "cp0-r1-v23-a-only-test",
                "--pre-dir",
                "/tmp",
                "--started",
                "1",
                "--plan",
                "0" * 64,
                "--mechanism-commands-sha256",
                "0" * 64,
                "--model-sha256",
                phone.MODEL_SHA256,
                "--histories",
                "/tmp/unused-histories",
                "--launch-plan",
                "/tmp/unused-plan",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"E_EXECUTION_NOT_CONFIRMED", completed.stderr)


if __name__ == "__main__":
    unittest.main()
