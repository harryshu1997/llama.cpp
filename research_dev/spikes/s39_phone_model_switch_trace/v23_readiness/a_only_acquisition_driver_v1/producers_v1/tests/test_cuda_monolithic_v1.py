#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest


HERE = Path(__file__).resolve().parent
PRODUCER = HERE.parent / "cuda_monolithic_v1.py"
SPEC = importlib.util.spec_from_file_location("cuda_monolithic_v1", PRODUCER)
cuda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cuda)
MODEL_SHA256 = "5" * 64
PHASE_ID = "cp0-r1-v23-a-only-cuda-monolithic-test"
PLAN_SHA256 = "7" * 64


def canonical(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def free_port() -> int:
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


WORKER_SOURCE = r'''#!/usr/bin/python3 -I
import argparse
import json
import os
import socket
import struct

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--model-sha256", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--backend", required=True)
parser.add_argument("--layer-start", required=True)
parser.add_argument("--layer-end", required=True)
parser.add_argument(
    "--test-mode",
    choices=(
        "good",
        "bad-lineage",
        "bad-pid",
        "bad-start-ticks",
        "low-ubatch",
    ),
    default="good",
)
parser.add_argument("--mode", required=True)
args = parser.parse_args()

def process_start_ticks():
    fields = open("/proc/self/stat", encoding="ascii").read().split()
    return int(fields[21])

runtime_process = {
    "boot_id": open(
        "/proc/sys/kernel/random/boot_id", encoding="ascii"
    ).read().strip(),
    "launcher_path": os.path.realpath("/proc/self/exe"),
    "loaded_repo_component_ids": [
        "cuda_monolithic.launcher",
        "cuda_monolithic.runtime",
    ],
    "pid": os.getpid() + (1000000 if args.test_mode == "bad-pid" else 0),
    "schema": "s39-runtime-process-source-v1",
    "start_ticks": process_start_ticks()
        + (1 if args.test_mode == "bad-start-ticks" else 0),
    "system_dependencies": [{
        "build_id": None,
        "ctime_ns": 10,
        "device_id": 11,
        "inode": 12,
        "mode": 33188,
        "mtime_ns": 13,
        "path": "/usr/lib/libc.so",
        "size": 14,
    }],
}
print(
    "RUNTIMEPROCESS "
    + json.dumps(
        runtime_process,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ),
    flush=True,
)

def recv_exact(connection, size):
    result = bytearray()
    while len(result) < size:
        block = connection.recv(size - len(result))
        if not block:
            raise RuntimeError("EOF")
        result.extend(block)
    return bytes(result)

def recv_i32(connection, count):
    return struct.unpack(f"<{count}i", recv_exact(connection, count * 4))

def send_i32(connection, values):
    connection.sendall(struct.pack(f"<{len(values)}i", *values))

active = set()
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", args.port))
    listener.listen(1)
    connection, _ = listener.accept()
    with connection:
        while True:
            opcode = recv_i32(connection, 1)[0]
            if opcode == -8:
                send_i32(connection, [
                    0x4C535633, 3, 0, 40, 40, 5120, 8, 256, 64,
                    32 if args.test_mode == "low-ubatch" else 64, 0x7F,
                ])
            elif opcode == -13:
                send_i32(connection, [0x4C534944, 1, 15])
                connection.sendall(bytes.fromhex(args.model_sha256))
            elif opcode == -11:
                version = recv_i32(connection, 1)[0]
                send_i32(connection, [0, version, len(active), 8, 0])
            elif opcode == -9:
                version, count, width = recv_i32(connection, 3)
                if version != 3 or width != 0:
                    raise RuntimeError("bad batch")
                request_ids = struct.unpack(
                    f"<{count}q", recv_exact(connection, count * 8)
                )
                if any(request_id <= 0 for request_id in request_ids):
                    raise RuntimeError("nonpositive request id")
                route_epochs = struct.unpack(
                    f"<{count}q", recv_exact(connection, count * 8)
                )
                seq_ids = struct.unpack(
                    f"<{count}i", recv_exact(connection, count * 4)
                )
                positions = struct.unpack(
                    f"<{count}i", recv_exact(connection, count * 4)
                )
                tokens = struct.unpack(
                    f"<{count}i", recv_exact(connection, count * 4)
                )
                active.update(seq_ids)
                if args.test_mode == "bad-lineage":
                    request_ids = (request_ids[0] + 1, *request_ids[1:])
                send_i32(connection, [0, count, 0])
                connection.sendall(struct.pack(f"<{count}q", *request_ids))
                connection.sendall(struct.pack(f"<{count}q", *route_epochs))
                connection.sendall(struct.pack(f"<{count}i", *seq_ids))
                connection.sendall(struct.pack(f"<{count}i", *positions))
                send_i32(connection, [token + 1 for token in tokens])
            elif opcode == -10:
                version, seq_id = recv_i32(connection, 2)
                recv_exact(connection, 16)
                active.remove(seq_id)
                send_i32(connection, [0, version, len(active), 8, 0])
            elif opcode == -1:
                break
            else:
                raise RuntimeError(f"bad opcode {opcode}")
'''

MANAGER_SOURCE = r'''#!/usr/bin/python3 -I
import argparse
import hashlib
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--plan-json", required=True)
parser.add_argument("--plan-sha256", required=True)
args = parser.parse_args()
raw = args.plan_json.encode("ascii")
if hashlib.sha256(raw).hexdigest() != args.plan_sha256:
    raise SystemExit(2)
plan = json.loads(args.plan_json)
route = plan["route"]
os.chdir(route["cwd"])
os.execve(route["argv"][0], route["argv"], route["environment"])
'''


class ProducerFixture:
    def __init__(self, test: unittest.TestCase, mode: str = "good"):
        self.temporary = tempfile.TemporaryDirectory()
        test.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.pre = self.root / "pre"
        self.pre.mkdir()
        self.output = self.root / "result.json"
        self.worker = self.root / "worker.py"
        self.worker.write_text(textwrap.dedent(WORKER_SOURCE), encoding="ascii")
        self.worker.chmod(0o755)
        self.manager = self.root / "manager.py"
        self.manager.write_text(textwrap.dedent(MANAGER_SOURCE), encoding="ascii")
        self.manager.chmod(0o755)
        self.port = free_port()
        self.phase_lock = {
            "phase": "A_ONLY",
            "phase_id": PHASE_ID,
        }
        (self.pre / "phase_lock.jsonl").write_bytes(canonical(self.phase_lock))
        self.histories = self.root / "histories.json"
        self.histories.write_bytes(canonical({
            "histories": [
                [10 + sequence * 10 + offset for offset in range(8)]
                for sequence in range(8)
            ],
            "history_width": 8,
            "model_id": "qwen3-14b-q4_k_m",
            "model_sha256": MODEL_SHA256,
            "request_ids": list(range(8)),
            "route_epoch": 11,
            "schema": "s39-cp0-r1-a-only-b8-histories-v1",
        }))
        self.launch = self.root / "launch.json"
        target = [
            sys.executable,
            "-I",
            "-B",
            str(self.worker),
            "--model",
            cuda.MODEL_PATH,
            "--mode",
            "monov3",
            "--backend",
            "CUDA0",
            "--layer-start",
            "0",
            "--layer-end",
            "40",
            "--port",
            str(self.port),
            "--model-sha256",
            MODEL_SHA256,
            "--test-mode",
            mode,
        ]
        environment = {
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        managed_plan = {
            "android": None,
            "bundle_id": "cuda_monolithic",
            "components": [{
                "component_id": "cuda_monolithic.launcher",
                "path": target[0],
            }],
            "endpoint": "cuda",
            "launcher_component_id": "cuda_monolithic.launcher",
            "mode": "local_cuda",
            "route": {
                "argv": target,
                "cwd": str(self.root),
                "environment": environment,
                "kind": "local_exec",
            },
            "schema": "s39-managed-runtime-launch-plan-v1",
        }
        managed_raw = canonical(managed_plan)[:-1].decode("ascii")
        command = [
            str(self.manager),
            "--plan-json",
            managed_raw,
            "--plan-sha256",
            hashlib.sha256(managed_raw.encode("ascii")).hexdigest(),
        ]
        self.mechanism_commands = {
            "desktop": [
                ["/token-codec"],
                ["/cuda-route"],
                ["/nvidia-device", "before"],
                ["/nvidia-process", "before"],
                ["/nvidia-device", "ready"],
                ["/nvidia-process", "ready"],
                ["/nvidia-device", "after"],
                ["/nvidia-process", "after"],
                command,
            ],
            "op12": [["/op12-worker"]],
            "op15": [["/op15-worker"], ["/op15-relay"]],
        }
        self.mechanism_sha256 = hashlib.sha256(
            canonical(self.mechanism_commands)
        ).hexdigest()
        self.launch.write_bytes(canonical({
            "command": command,
            "cwd": str(self.root),
            "expected_capabilities": 0x7F,
            "env": environment,
            "expected_file_type": 15,
            "expected_max_streams": 8,
            "expected_n_batch": 64,
            "expected_n_ctx_seq": 256,
            "expected_n_embd": 5120,
            "expected_n_layer": 40,
            "expected_n_ubatch": 64,
            "host": "127.0.0.1",
            "io_timeout_ms": 5000,
            "mechanism_commands": self.mechanism_commands,
            "mechanism_commands_sha256": self.mechanism_sha256,
            "model_id": "qwen3-14b-q4_k_m",
            "model_sha256": MODEL_SHA256,
            "port": self.port,
            "schema": "s39-cp0-r1-a-only-cuda-monolithic-launch-v1",
            "shutdown_timeout_ms": 5000,
            "startup_timeout_ms": 5000,
        }))

    def argv(self) -> list[str]:
        return [
            sys.executable,
            "-I",
            "-B",
            str(PRODUCER),
            "--output",
            str(self.output),
            "--phase-id",
            PHASE_ID,
            "--pre-dir",
            str(self.pre),
            "--started",
            "1",
            "--plan",
            PLAN_SHA256,
            "--mechanism-commands-sha256",
            self.mechanism_sha256,
            "--model-sha256",
            MODEL_SHA256,
            "--histories",
            str(self.histories),
            "--launch-plan",
            str(self.launch),
        ]

    def run(self) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            self.argv(),
            capture_output=True,
            check=False,
            timeout=20,
        )


class CudaMonolithicProducerTests(unittest.TestCase):
    def test_managed_command_exact_binds_pinned_model_path(self):
        environment = {
            "LAYERSPLIT_MEMORY_CERT": "1",
            "LAYERSPLIT_MODEL_SHA256": MODEL_SHA256,
            "LAYERSPLIT_PLACEMENT_CERT": "1",
        }
        target = [
            "/runtime/llama-layersplit",
            "--model",
            cuda.MODEL_PATH,
            "--mode",
            "monov3",
            "--backend",
            "CUDA0",
            "--layer-start",
            "0",
            "--layer-end",
            "40",
        ]
        plan = {
            "android": None,
            "bundle_id": "cuda_monolithic",
            "components": [{
                "component_id": "cuda_monolithic.launcher",
                "path": target[0],
            }],
            "endpoint": "cuda",
            "launcher_component_id": "cuda_monolithic.launcher",
            "mode": "local_cuda",
            "route": {
                "argv": target,
                "cwd": "/runtime",
                "environment": environment,
                "kind": "local_exec",
            },
            "schema": "s39-managed-runtime-launch-plan-v1",
        }
        inline = canonical(plan)[:-1].decode("ascii")
        command = [
            "/launcher",
            "--plan-json",
            inline,
            "--plan-sha256",
            hashlib.sha256(inline.encode("ascii")).hexdigest(),
        ]
        self.assertEqual(
            cuda.validate_managed_command(command, environment),
            target,
        )
        plan["route"]["argv"][2] = "/wrong.gguf"
        changed = canonical(plan)[:-1].decode("ascii")
        command[2] = changed
        command[4] = hashlib.sha256(changed.encode("ascii")).hexdigest()
        with self.assertRaises(cuda.CaptureError):
            cuda.validate_managed_command(command, environment)

    def test_real_protocol_fake_worker_emits_exact_b8_rows(self):
        fixture = ProducerFixture(self)
        process = fixture.run()
        self.assertEqual(process.returncode, 0, process.stderr.decode())
        self.assertEqual(process.stdout, b"")
        result = json.loads(fixture.output.read_bytes())
        self.assertEqual(
            result["schema"],
            "s39-cp0-r1-a-only-cuda-monolithic-raw-v1",
        )
        self.assertEqual(
            result["mechanism_commands_sha256"],
            fixture.mechanism_sha256,
        )
        rows = result["oracle_cuda_monolithic_rows"]
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[0]["state_count_before"], 0)
        self.assertEqual(rows[0]["state_count_after"], 0)
        self.assertEqual(
            rows[0]["call_shapes"],
            [
                {
                    "call_index": 0,
                    "n_seqs": 8,
                    "n_tokens": 64,
                    "phase": "prefill",
                },
                *[
                    {
                        "call_index": index,
                        "n_seqs": 8,
                        "n_tokens": 8,
                        "phase": "decode",
                    }
                    for index in range(1, 9)
                ],
            ],
        )
        for request_id, row in enumerate(rows[1:]):
            self.assertEqual(row["request_id"], request_id)
            self.assertEqual(row["positions"], list(range(8)))
            self.assertEqual(len(row["input_tokens"]), 8)
            self.assertEqual(len(row["continuation_tokens"]), 8)
            self.assertEqual(row["owner_before"], "CUDA")
            self.assertEqual(row["owner_after"], "RELEASED")
        runtime = result["runtime_process"]
        self.assertEqual(runtime["bundle_id"], "cuda_monolithic")
        self.assertEqual(runtime["endpoint"], "cuda")
        self.assertGreater(runtime["pid"], 0)
        self.assertGreater(runtime["start_ticks"], 0)
        self.assertGreater(runtime["observed_ns"], result["started_ns"])
        self.assertTrue(Path(runtime["launcher_path"]).is_absolute())
        self.assertEqual(
            runtime["loaded_repo_component_ids"],
            [
                "cuda_monolithic.launcher",
                "cuda_monolithic.runtime",
            ],
        )

    def test_lineage_mismatch_refuses_without_result(self):
        fixture = ProducerFixture(self, mode="bad-lineage")
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_BATCH_LINEAGE", process.stderr)

    def test_non_b8_histories_refuse_before_launch(self):
        fixture = ProducerFixture(self)
        value = json.loads(fixture.histories.read_bytes())
        value["histories"].pop()
        fixture.histories.write_bytes(canonical(value))
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_HISTORY_BATCH", process.stderr)

    def test_worker_ubatch_below_64_refuses(self):
        fixture = ProducerFixture(self, mode="low-ubatch")
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_HELLO_BATCH", process.stderr)

    def test_self_reported_pid_must_match_spawned_process(self):
        fixture = ProducerFixture(self, mode="bad-pid")
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_RUNTIME_PROCESS_PID", process.stderr)

    def test_self_reported_start_ticks_must_match_live_proc(self):
        fixture = ProducerFixture(self, mode="bad-start-ticks")
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_RUNTIME_PROCESS_START_TICKS", process.stderr)

    def test_noncanonical_histories_refuse_before_launch(self):
        fixture = ProducerFixture(self)
        fixture.histories.write_text(
            '{"schema":"s39-cp0-r1-a-only-b8-histories-v1",'
            '"schema":"duplicate"}\n',
            encoding="ascii",
        )
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_DUPLICATE_KEY", process.stderr)

    def test_existing_output_refuses_before_launch(self):
        fixture = ProducerFixture(self)
        fixture.output.write_bytes(b"occupied")
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(fixture.output.read_bytes(), b"occupied")
        self.assertIn(b"E_OUTPUT", process.stderr)

    def test_mechanism_digest_must_match_launch_plan(self):
        fixture = ProducerFixture(self)
        argv = fixture.argv()
        argv[argv.index("--mechanism-commands-sha256") + 1] = "0" * 64
        process = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=20,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_MECHANISM_COMMANDS_SHA256", process.stderr)

    def test_mechanism_matrix_is_content_bound(self):
        fixture = ProducerFixture(self)
        value = json.loads(fixture.launch.read_bytes())
        value["mechanism_commands"]["desktop"][1].append("--changed")
        fixture.launch.write_bytes(canonical(value))
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_MECHANISM_COMMANDS_CONTENT", process.stderr)

    def test_monolithic_command_must_occupy_frozen_matrix_slot(self):
        fixture = ProducerFixture(self)
        value = json.loads(fixture.launch.read_bytes())
        value["mechanism_commands"]["desktop"][8] = ["/other-monolithic"]
        value["mechanism_commands_sha256"] = hashlib.sha256(
            canonical(value["mechanism_commands"])
        ).hexdigest()
        fixture.launch.write_bytes(canonical(value))
        argv = fixture.argv()
        argv[argv.index("--mechanism-commands-sha256") + 1] = value[
            "mechanism_commands_sha256"
        ]
        process = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=20,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_MECHANISM_MONOLITHIC_COMMAND", process.stderr)

    def test_program_digest_binds_launch_plan(self):
        first = ProducerFixture(self)
        result = first.run()
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        first_digest = json.loads(first.output.read_bytes())[
            "oracle_cuda_monolithic_rows"
        ][0]["program_sha256"]

        second = ProducerFixture(self)
        value = json.loads(second.launch.read_bytes())
        value["env"]["PROBE_ID"] = "second"
        command = value["command"]
        managed_plan = json.loads(command[2])
        managed_plan["route"]["environment"] = value["env"]
        managed_raw = canonical(managed_plan)[:-1].decode("ascii")
        command[2] = managed_raw
        command[4] = hashlib.sha256(managed_raw.encode("ascii")).hexdigest()
        value["mechanism_commands"]["desktop"][8] = command
        second.mechanism_sha256 = hashlib.sha256(
            canonical(value["mechanism_commands"])
        ).hexdigest()
        value["mechanism_commands_sha256"] = second.mechanism_sha256
        second.launch.write_bytes(canonical(value))
        result = second.run()
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        second_digest = json.loads(second.output.read_bytes())[
            "oracle_cuda_monolithic_rows"
        ][0]["program_sha256"]
        self.assertNotEqual(first_digest, second_digest)

    def test_missing_runtime_process_refuses(self):
        fixture = ProducerFixture(self)
        source = fixture.worker.read_text(encoding="ascii")
        begin = source.index("runtime_process = {")
        end = source.index("\n\ndef recv_exact", begin)
        fixture.worker.write_text(
            source[:begin] + source[end + 2:],
            encoding="ascii",
        )
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_RUNTIME_PROCESS", process.stderr)

    def test_duplicate_runtime_process_refuses(self):
        fixture = ProducerFixture(self)
        source = fixture.worker.read_text(encoding="ascii")
        marker = "def recv_exact(connection, size):"
        duplicate = (
            "print(\"RUNTIMEPROCESS \" + json.dumps("
            "runtime_process,ensure_ascii=True,sort_keys=True,"
            "separators=(\",\",\":\")),flush=True)\n\n"
        )
        fixture.worker.write_text(
            source.replace(marker, duplicate + marker),
            encoding="ascii",
        )
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_RUNTIME_PROCESS_COUNT", process.stderr)


if __name__ == "__main__":
    unittest.main()
