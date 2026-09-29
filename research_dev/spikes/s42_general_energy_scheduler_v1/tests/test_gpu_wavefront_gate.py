#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest


HERE = Path(__file__).resolve().parent
GATE_DIR = (
    HERE.parent
    / "full_fp16_burstgpt_v1"
    / "gpu_wavefront_v1"
)
sys.path.insert(0, str(GATE_DIR))

from gpu_wavefront_gate import (  # noqa: E402
    FENCE_BEGIN,
    FENCE_DONE,
    FENCE_MAGIC,
    FENCE_REQUEST_STRUCT,
    FENCE_RESPONSE_STRUCT,
    FENCE_VERSION,
    LM_EXECUTE_REQUEST,
    LM_EXECUTE_REQUEST_STRUCT,
    LM_EXECUTE_RESPONSE,
    LM_EXECUTE_RESPONSE_STRUCT,
    LM_FLAG_F16_IO,
    LM_HELLO_REQUEST,
    LM_HELLO_REQUEST_STRUCT,
    LM_HELLO_RESPONSE,
    LM_HELLO_RESPONSE_STRUCT,
    LM_MAGIC,
    LM_VERSION,
    PROFILE_SCHEMA,
    WavefrontGate,
    WavefrontProfile,
    fnv32,
    recv_exact,
)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def profile_json(*, saving: bool = True) -> dict[str, object]:
    avoided = 1000 if saving else 200
    return {
        "schema": PROFILE_SCHEMA,
        "profile_id": "wavefront-gate-test",
        "admission": "mechanics",
        "model": {
            "sha256": "1" * 64,
            "lm_head_weight_sha256": "2" * 64,
            "lm_head_resident_bytes": 100,
        },
        "memory": {
            "gpu_uuid": "GPU-test",
            "total_bytes": 10_000,
            "free_bytes": 2_000,
            "reserve_bytes": 1_000,
        },
        "worker": {
            "n_embd": 4,
            "n_vocab": 8,
            "offset": 1,
            "rows": 7,
            "top_k": 1,
            "flags": LM_FLAG_F16_IO,
            "weight_hash64": "0000000000000123",
        },
        "bubble": {
            "protected_ready_lower_us": 500_000,
            "guard_us": 10_000,
            "runtime_verified": False,
        },
        "candidate": {
            "workspace_bytes": 1,
            "service_latency_us": {
                "mean": 50_000,
                "upper": 100_000,
                "lower": 10_000,
                "sample_count": 0,
                "measured": False,
            },
            "restore_latency_us": {
                "mean": 1_000,
                "upper": 2_000,
                "lower": 500,
                "sample_count": 0,
                "measured": False,
            },
            "avoided_energy_uj": {
                "mean": avoided,
                "upper": avoided + 100,
                "lower": avoided - 100,
                "sample_count": 0,
                "measured": False,
            },
            "backfill_energy_uj": {
                "mean": 300,
                "upper": 400,
                "lower": 200,
                "sample_count": 0,
                "measured": False,
            },
            "energy_boundary_id": "cpu-package+gpu-board+whole-phone",
            "accounting_scope": "fp16-burstgpt-74",
            "minimum_energy_saving_ppm": 50_000,
        },
        "producer_resource_ids": ["cpu"],
        "evidence_ids": ["wavefront-gate-test"],
        "valid_for_us": 10_000_000,
    }


class FakeWorker:
    def __init__(self, port: int):
        self.port = port
        self.calls = 0
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.error: BaseException | None = None

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(2):
            raise RuntimeError("fake worker did not start")

    def run(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", self.port))
                listener.listen(1)
                self.ready.set()
                connection, _ = listener.accept()
                with connection:
                    hello = recv_exact(connection, LM_HELLO_REQUEST_STRUCT.size)
                    if hello is None:
                        raise RuntimeError("fake worker hello is absent")
                    connection.sendall(LM_HELLO_RESPONSE_STRUCT.pack(
                        LM_MAGIC,
                        LM_VERSION,
                        LM_HELLO_RESPONSE,
                        0,
                        LM_FLAG_F16_IO,
                        4,
                        8,
                        1,
                        7,
                        1,
                        1,
                        0x123,
                    ))
                    while True:
                        header = recv_exact(
                            connection,
                            LM_EXECUTE_REQUEST_STRUCT.size,
                            eof_ok=True,
                        )
                        if header is None:
                            break
                        payload = recv_exact(connection, 8)
                        if payload is None:
                            raise RuntimeError("fake worker payload is absent")
                        request = LM_EXECUTE_REQUEST_STRUCT.unpack(header)
                        candidate = b"\x03\x00\x00\x00\x00\x00\x80?"
                        connection.sendall(LM_EXECUTE_RESPONSE_STRUCT.pack(
                            LM_MAGIC,
                            LM_VERSION,
                            LM_EXECUTE_RESPONSE,
                            0,
                            0,
                            request[3],
                            1,
                            len(candidate),
                            fnv32(candidate),
                            1000,
                            100,
                        ) + candidate)
                        self.calls += 1
        except BaseException as exc:  # pylint: disable=broad-except
            self.error = exc


class GateFixture:
    def __init__(self, root: Path, *, mode: str, saving: bool = True):
        self.root = root
        self.worker_port = free_port()
        self.gate_port = free_port()
        self.profile_path = root / "profile.json"
        self.profile_path.write_text(
            json.dumps(profile_json(saving=saving)), encoding="ascii"
        )
        self.worker = FakeWorker(self.worker_port)
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
            output=root / "result.json",
            pipeline_id="gemma-request-test",
            max_backfills=1 if mode == "mechanics" else 0,
            connect_timeout_s=2.0,
            timeout_s=3.0,
        )
        self.gate = WavefrontGate(
            self.args, WavefrontProfile.load(self.profile_path)
        )
        self.gate_error: BaseException | None = None
        self.gate_thread = threading.Thread(target=self.run_gate, daemon=True)
        self.gate_thread.start()
        deadline = time.monotonic() + 2
        while not self.args.fence_socket.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("gate did not start")
            time.sleep(0.005)
        self.client = socket.create_connection(
            ("127.0.0.1", self.gate_port), timeout=2
        )
        self.client.settimeout(2)
        self.client.sendall(LM_HELLO_REQUEST_STRUCT.pack(
            LM_MAGIC,
            LM_VERSION,
            LM_HELLO_REQUEST,
            4,
            7,
            1,
            LM_FLAG_F16_IO,
            0,
        ))
        hello = recv_exact(self.client, LM_HELLO_RESPONSE_STRUCT.size)
        if hello is None:
            raise RuntimeError("gate hello is absent")

    def run_gate(self) -> None:
        try:
            self.gate.run()
        except BaseException as exc:  # pylint: disable=broad-except
            self.gate_error = exc

    def execute(self, request_id: int) -> bytes:
        payload = bytes(range(8))
        self.client.sendall(LM_EXECUTE_REQUEST_STRUCT.pack(
            LM_MAGIC,
            LM_VERSION,
            LM_EXECUTE_REQUEST,
            request_id,
            4,
            len(payload),
            fnv32(payload),
        ) + payload)
        result = recv_exact(
            self.client, LM_EXECUTE_RESPONSE_STRUCT.size + 8
        )
        if result is None:
            raise RuntimeError("LM-head result is absent")
        return result

    def fence_once(self, request_id: int = 9) -> tuple[int, ...]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.connect(str(self.args.fence_socket))
            connection.sendall(FENCE_REQUEST_STRUCT.pack(
                FENCE_MAGIC,
                FENCE_VERSION,
                FENCE_BEGIN,
                1,
                request_id,
                0,
                1,
                0,
                time.monotonic_ns(),
            ))
            response = connection.recv(FENCE_RESPONSE_STRUCT.size)
            return FENCE_RESPONSE_STRUCT.unpack(response)

    def finish(self) -> None:
        self.args.qwen_complete_file.write_text(
            f"{time.monotonic_ns()}\n", encoding="ascii"
        )
        self.client.close()
        self.gate_thread.join(3)
        self.gate.close()
        self.worker.thread.join(2)
        if self.gate_thread.is_alive():
            raise RuntimeError("gate did not stop")
        if self.gate_error is not None:
            raise self.gate_error
        if self.worker.error is not None:
            raise self.worker.error


class GpuWavefrontGateTests(unittest.TestCase):
    def test_control_holds_measured_work_until_qwen_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(Path(directory), mode="control")
            fixture.args.arm_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            result: list[bytes] = []
            request = threading.Thread(
                target=lambda: result.append(fixture.execute(1)), daemon=True
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
            fixture.client.close()
            fixture.gate_thread.join(3)
            fixture.gate.close()
            fixture.worker.thread.join(2)
            self.assertIsNone(fixture.gate_error)
            self.assertEqual(fixture.gate.backfills, 0)
            self.assertEqual(fixture.gate.tail_calls, 1)
            self.assertTrue(fixture.args.paid_tail_file.is_file())

    def test_mechanics_executes_one_scheduler_admitted_wavefront(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(Path(directory), mode="mechanics")
            fixture.args.arm_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            result: list[bytes] = []
            request = threading.Thread(
                target=lambda: result.append(fixture.execute(1)), daemon=True
            )
            request.start()
            deadline = time.monotonic() + 2
            while fixture.gate.peek_pending() is None:
                if time.monotonic() >= deadline:
                    self.fail("LM-head input did not become ready")
                time.sleep(0.005)
            response = fixture.fence_once()
            request.join(2)
            self.assertEqual(response[:3], (FENCE_MAGIC, FENCE_VERSION, FENCE_DONE))
            self.assertEqual(len(result), 1)
            self.assertEqual(fixture.gate.backfills, 1)
            self.assertEqual(len(fixture.gate.completed_receipts), 1)
            fixture.finish()

    def test_energy_rejection_falls_back_to_post_qwen_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(
                Path(directory), mode="mechanics", saving=False
            )
            fixture.args.arm_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            result: list[bytes] = []
            request = threading.Thread(
                target=lambda: result.append(fixture.execute(1)), daemon=True
            )
            request.start()
            deadline = time.monotonic() + 2
            while fixture.gate.peek_pending() is None:
                if time.monotonic() >= deadline:
                    self.fail("LM-head input did not become ready")
                time.sleep(0.005)
            fixture.fence_once()
            self.assertEqual(result, [])
            self.assertEqual(fixture.gate.backfills, 0)
            self.assertEqual(fixture.gate.rejections, 1)
            fixture.args.qwen_complete_file.write_text(
                f"{time.monotonic_ns()}\n", encoding="ascii"
            )
            request.join(2)
            self.assertEqual(len(result), 1)
            fixture.client.close()
            fixture.gate_thread.join(3)
            fixture.gate.close()
            fixture.worker.thread.join(2)
            self.assertIsNone(fixture.gate_error)
            self.assertEqual(fixture.gate.tail_calls, 1)


if __name__ == "__main__":
    unittest.main()
