#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest


HERE = Path(__file__).resolve().parent
GATE_DIR = HERE.parent / "full_fp16_burstgpt_v1" / "gpu_wavefront_v1"
sys.path.insert(0, str(GATE_DIR))

from gpu_stage_wavefront_gate import (  # noqa: E402
    DECODE_HEADER_STRUCT,
    FENCE_BEGIN,
    FENCE_DONE,
    FENCE_MAGIC,
    FENCE_REQUEST_STRUCT,
    FENCE_RESPONSE_STRUCT,
    FENCE_VERSION,
    I32,
    IDENTITY_STRUCT,
    LEGACY_HELLO_STRUCT,
    PREFILL_HEADER_STRUCT,
    PROFILE_SCHEMA,
    RESPONSE_HEADER_STRUCT,
    STAGE_BATCH_DECODE,
    STAGE_BATCH_PREFILL,
    STAGE_DETACH,
    STAGE_HELLO,
    STAGE_IDENTITY_MAGIC,
    STAGE_IDENTITY_VERSION,
    STAGE_RESET,
    STAGE_STOP,
    STAGE_V3_BATCH,
    STAGE_V3_HELLO,
    STAGE_V3_IDENTITY,
    STAGE_V3_MAGIC,
    STAGE_V3_VERSION,
    StageWavefrontGate,
    StageWavefrontProfile,
    V3_BATCH_HEADER_STRUCT,
    V3_HELLO_STRUCT,
    V3_RESPONSE_HEADER_STRUCT,
    recv_exact,
)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def metric(mean: int, upper: int, lower: int = 1) -> dict[str, object]:
    return {
        "lower": lower,
        "mean": mean,
        "measured": False,
        "sample_count": 0,
        "upper": upper,
    }


def profile_json(*, saving: bool = True) -> dict[str, object]:
    avoided = 1000 if saving else 200
    return {
        "schema": PROFILE_SCHEMA,
        "profile_id": "gpu-stage-gate-test",
        "admission": "mechanics",
        "model": {
            "sha256": "11" * 32,
            "stage_weight_sha256": "22" * 32,
            "resident_bytes": 100,
            "n_layer": 48,
            "n_embd": 4,
            "file_type": 1,
            "layer_start": 0,
            "layer_end": 1,
        },
        "memory": {
            "gpu_uuid": "GPU-test",
            "total_bytes": 10_000,
            "free_bytes": 2_000,
            "reserve_bytes": 1_000,
        },
        "bubble": {
            "protected_ready_lower_us": 500_000,
            "guard_us": 10_000,
            "runtime_verified": True,
        },
        "candidate": {
            "workspace_bytes": 1,
            "prefill_chunk_rows": 2,
            "prefill_service_latency_us": metric(50_000, 100_000),
            "decode_service_latency_us": metric(10_000, 20_000),
            "restore_latency_us": metric(1_000, 2_000),
            "avoided_energy_uj": metric(
                avoided, avoided + 100, max(1, avoided - 100)
            ),
            "backfill_energy_uj": metric(300, 400),
            "energy_boundary_id": "cpu-package+gpu-board+whole-phone",
            "accounting_scope": "fp16-burstgpt-stage-test",
            "minimum_energy_saving_ppm": 50_000,
        },
        "producer_resource_ids": ["cpu", "op15-htp"],
        "protected_resource_ids": ["op15-htp"],
        "fenced_resource_ids": [],
        "evidence_ids": ["gpu-stage-gate-test"],
        "valid_for_us": 10_000_000,
    }


class FakeStageWorker:
    def __init__(self, port: int):
        self.port = port
        self.calls = 0
        self.sessions = 0
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.error: BaseException | None = None

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(2):
            raise RuntimeError("fake stage worker did not start")

    @staticmethod
    def read_call(connection: socket.socket, command: int) -> tuple[int, bytes]:
        if command == STAGE_BATCH_PREFILL:
            rest = recv_exact(connection, PREFILL_HEADER_STRUCT.size - I32.size)
            if rest is None:
                raise RuntimeError("short prefill header")
            packet = I32.pack(command) + rest
            _, streams, tokens, width = PREFILL_HEADER_STRUCT.unpack(packet)
            rows = streams * tokens
            payload_size = rows * 4 + rows * width * 4
        else:
            rest = recv_exact(connection, DECODE_HEADER_STRUCT.size - I32.size)
            if rest is None:
                raise RuntimeError("short decode header")
            packet = I32.pack(command) + rest
            _, rows, width = DECODE_HEADER_STRUCT.unpack(packet)
            payload_size = rows * 12 + rows * width * 4
        payload = recv_exact(connection, payload_size)
        if payload is None:
            raise RuntimeError("short stage payload")
        return rows, packet + payload

    def run(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", self.port))
                listener.listen(2)
                self.ready.set()
                stopped = False
                while not stopped:
                    connection, _ = listener.accept()
                    self.sessions += 1
                    with connection:
                        while True:
                            raw = recv_exact(connection, I32.size, eof_ok=True)
                            if raw is None:
                                break
                            command = I32.unpack(raw)[0]
                            if command == STAGE_V3_HELLO:
                                connection.sendall(V3_HELLO_STRUCT.pack(
                                    STAGE_V3_MAGIC,
                                    STAGE_V3_VERSION,
                                    0,
                                    1,
                                    48,
                                    4,
                                    1,
                                    64,
                                    16,
                                    16,
                                    (1 << 0) | (1 << 1) | (1 << 2)
                                    | (1 << 3) | (1 << 5),
                                ))
                            elif command == STAGE_V3_IDENTITY:
                                connection.sendall(IDENTITY_STRUCT.pack(
                                    STAGE_IDENTITY_MAGIC,
                                    STAGE_IDENTITY_VERSION,
                                    1,
                                ) + bytes.fromhex("11" * 32))
                            elif command == STAGE_HELLO:
                                connection.sendall(LEGACY_HELLO_STRUCT.pack(
                                    0x4C53504C,
                                    1,
                                    0,
                                    1,
                                    48,
                                    4,
                                    1,
                                    64,
                                    16,
                                    16,
                                ))
                            elif command == STAGE_RESET:
                                connection.sendall(I32.pack(0))
                            elif command == STAGE_V3_BATCH:
                                rest = recv_exact(
                                    connection,
                                    V3_BATCH_HEADER_STRUCT.size - I32.size,
                                )
                                if rest is None:
                                    raise RuntimeError("short V3 batch header")
                                header = V3_BATCH_HEADER_STRUCT.unpack(raw + rest)
                                _, version, rows, width = header
                                if version != STAGE_V3_VERSION or width != 0:
                                    raise RuntimeError("bad V3 batch header")
                                vector_size = rows * (8 + 8 + 4 + 4 + 4)
                                vectors = recv_exact(connection, vector_size)
                                if vectors is None:
                                    raise RuntimeError("short V3 batch vectors")
                                identity_size = rows * (8 + 8 + 4 + 4)
                                output = b"".join(
                                    struct.pack("<f", float(index))
                                    for index in range(rows * 4)
                                )
                                connection.sendall(
                                    V3_RESPONSE_HEADER_STRUCT.pack(0, rows, 4)
                                    + vectors[:identity_size]
                                    + output
                                )
                                self.calls += 1
                            elif command in {STAGE_BATCH_PREFILL, STAGE_BATCH_DECODE}:
                                rows, _ = self.read_call(connection, command)
                                payload = b"".join(
                                    struct.pack("<f", float(index))
                                    for index in range(rows * 4)
                                )
                                connection.sendall(
                                    RESPONSE_HEADER_STRUCT.pack(rows, 4) + payload
                                )
                                self.calls += 1
                            elif command == STAGE_DETACH:
                                connection.sendall(I32.pack(0))
                                break
                            elif command == STAGE_STOP:
                                stopped = True
                                break
                            else:
                                raise RuntimeError(f"bad fake command {command}")
        except BaseException as exc:  # pylint: disable=broad-except
            self.error = exc


class GateFixture:
    def __init__(
        self,
        root: Path,
        *,
        mode: str,
        saving: bool = True,
        max_backfills: int | None = None,
        prepare: bool = False,
    ):
        self.root = root
        self.worker_port = free_port()
        self.gate_port = free_port()
        self.profile_path = root / "profile.json"
        self.profile_path.write_text(
            json.dumps(profile_json(saving=saving)), encoding="ascii"
        )
        self.worker = FakeStageWorker(self.worker_port)
        self.worker.start()
        self.args = argparse.Namespace(
            mode=mode,
            profile=self.profile_path,
            listen_host="127.0.0.1",
            listen_port=self.gate_port,
            worker_host="127.0.0.1",
            worker_port=self.worker_port,
            fence_socket=root / "fence.sock",
            arm_file=root / "arm.receipt",
            qwen_complete_file=root / "qwen.receipt",
            paid_tail_file=root / "tail.receipt",
            candidate_ready_file=root / "candidate.receipt",
            prepare_file=root / "prepare.receipt" if prepare else None,
            prepared_file=root / "prepared.receipt" if prepare else None,
            prepare_max_replays=4,
            prepare_required_consecutive=2,
            output=root / "result.json",
            pipeline_id="gemma-stage-test",
            max_backfills=(
                max_backfills
                if max_backfills is not None
                else (1 if mode == "mechanics" else 0)
            ),
            connect_timeout_s=2.0,
            timeout_s=3.0,
        )
        self.gate = StageWavefrontGate(
            self.args, StageWavefrontProfile.load(self.profile_path)
        )
        self.gate_error: BaseException | None = None
        self.gate_thread = threading.Thread(target=self.run_gate, daemon=True)
        self.gate_thread.start()
        self.fence_sequence = 0
        deadline = time.monotonic() + 2
        while not self.args.fence_socket.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("GPU-stage gate did not start")
            time.sleep(0.005)
        self.client = self.connect_client()

    def connect_client(self) -> socket.socket:
        client = socket.create_connection(("127.0.0.1", self.gate_port), timeout=2)
        client.settimeout(2)
        client.sendall(I32.pack(STAGE_HELLO))
        hello = recv_exact(client, LEGACY_HELLO_STRUCT.size)
        if hello is None:
            raise RuntimeError("stage hello is absent")
        client.sendall(I32.pack(STAGE_RESET))
        reset = recv_exact(client, I32.size)
        if reset is None or I32.unpack(reset)[0] != 0:
            raise RuntimeError("stage reset failed")
        return client

    def run_gate(self) -> None:
        try:
            self.gate.run()
        except BaseException as exc:  # pylint: disable=broad-except
            self.gate_error = exc

    def execute_prefill(self, tokens: tuple[int, ...] = (3, 4)) -> bytes:
        packet = PREFILL_HEADER_STRUCT.pack(
            STAGE_BATCH_PREFILL, 1, len(tokens), 0
        )
        packet += struct.pack(f"<{len(tokens)}i", *tokens)
        self.client.sendall(packet)
        result = recv_exact(
            self.client,
            RESPONSE_HEADER_STRUCT.size + len(tokens) * 4 * 4,
        )
        if result is None:
            raise RuntimeError("stage result is absent")
        return result

    def fence_once(self) -> tuple[int, ...]:
        self.fence_sequence += 1
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.connect(str(self.args.fence_socket))
            connection.sendall(FENCE_REQUEST_STRUCT.pack(
                FENCE_MAGIC,
                FENCE_VERSION,
                FENCE_BEGIN,
                self.fence_sequence,
                8 + self.fence_sequence,
                0,
                1,
                0,
                time.monotonic_ns(),
            ))
            response = connection.recv(FENCE_RESPONSE_STRUCT.size)
            return FENCE_RESPONSE_STRUCT.unpack(response)

    def detach_and_reconnect(self) -> None:
        self.client.sendall(I32.pack(STAGE_DETACH))
        response = recv_exact(self.client, I32.size)
        if response is None or I32.unpack(response)[0] != 0:
            raise RuntimeError("stage detach failed")
        self.client.close()
        deadline = time.monotonic() + 2
        while self.gate.client_thread is not None and self.gate.client_thread.is_alive():
            if time.monotonic() >= deadline:
                raise RuntimeError("stage client did not detach")
            time.sleep(0.005)
        self.client = self.connect_client()

    def stop(self) -> None:
        self.client.sendall(I32.pack(STAGE_STOP))
        self.client.close()
        if not self.args.qwen_complete_file.exists():
            self.args.qwen_complete_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
        self.gate_thread.join(3)
        self.gate.close()
        self.worker.thread.join(2)
        if self.gate_thread.is_alive():
            raise RuntimeError("GPU-stage gate did not stop")
        if self.gate_error is not None:
            raise self.gate_error
        if self.worker.error is not None:
            raise self.worker.error


class GpuStageWavefrontGateTests(unittest.TestCase):
    def test_warmup_detach_reconnects_same_resident_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(Path(directory), mode="mechanics")
            warmup = fixture.execute_prefill()
            self.assertTrue(warmup)
            self.assertEqual(fixture.gate.warmup_calls, 1)
            fixture.detach_and_reconnect()
            fixture.args.arm_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            result: list[bytes] = []
            request = threading.Thread(
                target=lambda: result.append(fixture.execute_prefill()), daemon=True
            )
            request.start()
            deadline = time.monotonic() + 2
            while fixture.gate.peek_pending() is None:
                if time.monotonic() >= deadline:
                    self.fail("stage call did not become ready after detach")
                time.sleep(0.005)
            fixture.fence_once()
            request.join(2)
            self.assertEqual(len(result), 1)
            self.assertEqual(fixture.gate.backfills, 1)
            self.assertEqual(fixture.gate.session_index, 1)
            fixture.stop()

    def test_streamed_prefill_calls_keep_cumulative_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(Path(directory), mode="mechanics")
            fixture.execute_prefill((3, 4))
            fixture.execute_prefill((5, 6))
            events = [
                event
                for event in fixture.gate.events
                if event.get("event") == "worker_execute"
            ]
            self.assertEqual(
                [event["position_start"] for event in events],
                [0, 2],
            )
            self.assertEqual(fixture.gate.prefill_position, 4)
            fixture.stop()

    def test_control_holds_stage_until_qwen_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(Path(directory), mode="control")
            fixture.args.arm_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            result: list[bytes] = []
            request = threading.Thread(
                target=lambda: result.append(fixture.execute_prefill()), daemon=True
            )
            request.start()
            time.sleep(0.05)
            response = fixture.fence_once()
            self.assertEqual(response[:3], (FENCE_MAGIC, FENCE_VERSION, FENCE_DONE))
            self.assertEqual(result, [])
            fixture.args.qwen_complete_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            request.join(2)
            self.assertEqual(len(result), 1)
            self.assertEqual(fixture.gate.tail_calls, 1)
            fixture.stop()

    def test_adaptive_mode_rejects_shared_htp_contention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(Path(directory), mode="adaptive")
            fixture.args.arm_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            result: list[bytes] = []
            request = threading.Thread(
                target=lambda: result.append(fixture.execute_prefill()), daemon=True
            )
            request.start()
            deadline = time.monotonic() + 2
            while fixture.gate.peek_pending() is None:
                if time.monotonic() >= deadline:
                    self.fail("adaptive stage call did not become ready")
                time.sleep(0.005)
            fixture.fence_once()
            self.assertEqual(result, [])
            self.assertEqual(fixture.gate.backfills, 0)
            self.assertEqual(fixture.gate.rejections, 1)
            rejection = next(
                event for event in fixture.gate.events
                if event.get("event") == "wavefront_rejected"
            )
            self.assertEqual(
                rejection["decision"]["backfill"]["rejected"][0]["reason"],
                "RESOURCE_NOT_READY",
            )
            self.assertEqual(
                {
                    lease["resource_id"]
                    for lease in fixture.gate.protected_resource_leases
                },
                {"op15-htp"},
            )
            fixture.args.qwen_complete_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            request.join(2)
            self.assertEqual(len(result), 1)
            fixture.stop()

    def test_mechanics_schedules_ready_stage_prefill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(Path(directory), mode="mechanics")
            fixture.args.arm_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            result: list[bytes] = []
            request = threading.Thread(
                target=lambda: result.append(fixture.execute_prefill()), daemon=True
            )
            request.start()
            deadline = time.monotonic() + 2
            while fixture.gate.peek_pending() is None:
                if time.monotonic() >= deadline:
                    self.fail("stage call did not become ready")
                time.sleep(0.005)
            response = fixture.fence_once()
            request.join(2)
            self.assertEqual(response[:3], (FENCE_MAGIC, FENCE_VERSION, FENCE_DONE))
            self.assertEqual(len(result), 1)
            self.assertEqual(fixture.gate.backfills, 1)
            self.assertEqual(fixture.gate.tail_calls, 0)
            fixture.stop()

    def test_envelope_overrun_trips_recoverable_circuit_breaker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(Path(directory), mode="mechanics")
            fixture.args.arm_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            assert fixture.gate.scheduler is not None
            original_release = (
                fixture.gate.scheduler.release_gpu_wavefront_backfill
            )

            def reject_release(*_args: object) -> None:
                raise RuntimeError(
                    "GPU backfill completion is outside its decision envelope"
                )

            fixture.gate.scheduler.release_gpu_wavefront_backfill = reject_release
            result: list[bytes] = []
            request = threading.Thread(
                target=lambda: result.append(fixture.execute_prefill()), daemon=True
            )
            request.start()
            deadline = time.monotonic() + 2
            while fixture.gate.peek_pending() is None:
                if time.monotonic() >= deadline:
                    self.fail("circuit-breaker stage call did not become ready")
                time.sleep(0.005)
            fixture.fence_once()
            request.join(2)
            self.assertEqual(len(result), 1)
            self.assertEqual(fixture.gate.backfills, 0)
            self.assertEqual(fixture.gate.fallback_executions, 1)
            self.assertEqual(fixture.gate.circuit_breaker_at_backfill, 0)
            self.assertIn(
                "outside its decision envelope",
                fixture.gate.circuit_breaker_reason or "",
            )
            fixture.gate.scheduler.release_gpu_wavefront_backfill = original_release
            fixture.stop()

    def test_prefill_chunks_keep_request_blocked_until_causal_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(Path(directory), mode="mechanics")
            fixture.args.arm_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            result: list[bytes] = []
            request = threading.Thread(
                target=lambda: result.append(
                    fixture.execute_prefill((3, 4, 5, 6, 7))
                ),
                daemon=True,
            )
            request.start()
            deadline = time.monotonic() + 2
            while fixture.gate.peek_pending() is None:
                if time.monotonic() >= deadline:
                    self.fail("chunked stage call did not become ready")
                time.sleep(0.005)
            fixture.fence_once()
            self.assertEqual(result, [])
            self.assertEqual(fixture.gate.backfills, 1)
            self.assertEqual(fixture.gate.peek_pending().next_chunk, 1)
            fixture.args.qwen_complete_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            request.join(2)
            self.assertEqual(len(result), 1)
            self.assertEqual(
                RESPONSE_HEADER_STRUCT.unpack(
                    result[0][: RESPONSE_HEADER_STRUCT.size]
                ),
                (5, 4),
            )
            self.assertEqual(fixture.gate.chunk_calls, 3)
            self.assertEqual(fixture.gate.tail_calls, 2)
            fixture.stop()

    def test_prefill_chunks_backfill_causally_across_fences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(
                Path(directory), mode="mechanics", max_backfills=3
            )
            fixture.args.arm_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            result: list[bytes] = []
            request = threading.Thread(
                target=lambda: result.append(
                    fixture.execute_prefill((3, 4, 5, 6, 7))
                ),
                daemon=True,
            )
            request.start()
            deadline = time.monotonic() + 2
            while fixture.gate.peek_pending() is None:
                if time.monotonic() >= deadline:
                    self.fail("chunked stage call did not become ready")
                time.sleep(0.005)
            for _ in range(3):
                fixture.fence_once()
            request.join(2)
            self.assertEqual(len(result), 1)
            self.assertEqual(fixture.gate.backfills, 3)
            self.assertEqual(fixture.gate.tail_calls, 0)
            events = [
                event
                for event in fixture.gate.events
                if event.get("event") == "wavefront_execute"
            ]
            self.assertEqual(
                [event["chunk_index"] for event in events], [0, 1, 2]
            )
            fixture.stop()

    def test_candidate_preparation_replays_without_advancing_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(
                Path(directory), mode="mechanics", prepare=True
            )
            fixture.args.arm_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            result: list[bytes] = []
            request = threading.Thread(
                target=lambda: result.append(fixture.execute_prefill()), daemon=True
            )
            request.start()
            deadline = time.monotonic() + 2
            while not fixture.args.candidate_ready_file.exists():
                if time.monotonic() >= deadline:
                    self.fail("stage candidate did not become ready")
                time.sleep(0.005)
            self.assertIsNotNone(fixture.args.prepare_file)
            self.assertIsNotNone(fixture.args.prepared_file)
            fixture.args.prepare_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            deadline = time.monotonic() + 2
            while not fixture.args.prepared_file.exists():
                if time.monotonic() >= deadline:
                    self.fail("stage candidate was not prepared")
                time.sleep(0.005)
            prepare_calls = fixture.gate.prepare_calls
            self.assertGreaterEqual(prepare_calls, 2)
            self.assertLessEqual(prepare_calls, 4)
            self.assertEqual(
                len(fixture.gate.prepare_durations_us), prepare_calls
            )
            self.assertTrue(all(
                value <= 100_000
                for value in fixture.gate.prepare_durations_us[-2:]
            ))
            self.assertEqual(fixture.gate.peek_pending().next_chunk, 0)
            fixture.fence_once()
            request.join(2)
            self.assertEqual(len(result), 1)
            self.assertEqual(fixture.gate.backfills, 1)
            self.assertEqual(fixture.gate.prepare_calls, prepare_calls)
            fixture.stop()

    def test_energy_rejection_drains_after_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(Path(directory), mode="mechanics", saving=False)
            fixture.args.arm_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            result: list[bytes] = []
            request = threading.Thread(
                target=lambda: result.append(fixture.execute_prefill()), daemon=True
            )
            request.start()
            deadline = time.monotonic() + 2
            while fixture.gate.peek_pending() is None:
                if time.monotonic() >= deadline:
                    self.fail("stage call did not become ready")
                time.sleep(0.005)
            fixture.fence_once()
            self.assertEqual(result, [])
            self.assertEqual(fixture.gate.rejections, 1)
            fixture.args.qwen_complete_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            request.join(2)
            self.assertEqual(len(result), 1)
            fixture.stop()


if __name__ == "__main__":
    unittest.main()
