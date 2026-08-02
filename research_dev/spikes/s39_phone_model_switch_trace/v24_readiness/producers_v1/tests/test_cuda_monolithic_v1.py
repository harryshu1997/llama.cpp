#!/usr/bin/env python3

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
PRODUCER = HERE.parent / "cuda_monolithic_v1.py"
S39 = HERE.parents[2]
FROZEN_CORPUS = S39 / "CP0_R1_MMLU64_CORPUS_V2_2.jsonl"
PRODUCTION_ROOTS_SOURCE = '''ALLOWED_SYSTEM_ROOTS = [
    "/mnt/storage/s21_deps/cuda-13.2.1/lib/",
    "/usr/lib/x86_64-linux-gnu/",
]'''
TEST_ROOTS_SOURCE = '''ALLOWED_SYSTEM_ROOTS = [
    "/mnt/storage/s21_deps/cuda-13.2.1/lib/",
    "/usr/lib/python3.12/",
    "/usr/lib/x86_64-linux-gnu/",
]'''
PRODUCTION_MAP_SOURCE = '''EXPECTED_MODEL_MAP_OFFSETS = (
    0x26645000,
    0x2188BD000,
)
EXPECTED_MODEL_MAP_PERMISSIONS = "r--s"'''
TEST_MAP_SOURCE = '''EXPECTED_MODEL_MAP_OFFSETS = (
    0,
)
EXPECTED_MODEL_MAP_PERMISSIONS = "r--s"'''
MODEL_SHA256 = "5" * 64
PHASE_ID = "cp0-r1-v24-a-only-cuda-monolithic-test"
MECHANISM_SHA256 = "6" * 64
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


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_stat(path: Path) -> dict[str, int]:
    value = path.stat(follow_symlinks=False)
    return {
        "ctime_ns": value.st_ctime_ns,
        "device_id": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "mtime_ns": value.st_mtime_ns,
        "size": value.st_size,
    }


def free_port() -> int:
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


WORKER_SOURCE = r'''
import argparse
import json
import mmap
import os
import socket
import struct

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--model-sha256", required=True)
parser.add_argument("--fake-mode", default="good")
parser.add_argument("--extra-model")
parser.add_argument("--mode")
parser.add_argument("-m", dest="model", required=True)
parser.add_argument("--devices")
parser.add_argument("--driver-batch")
parser.add_argument("--driver-context")
parser.add_argument("--driver-max-prefill")
args = parser.parse_args()

model_file = open(args.model, "rb")
model_map = None
if args.fake_mode != "no-map":
    model_map = mmap.mmap(model_file.fileno(), 0, access=mmap.ACCESS_READ)
extra_file = None
extra_map = None
if args.extra_model:
    extra_file = open(args.extra_model, "rb")
    extra_map = mmap.mmap(extra_file.fileno(), 0, access=mmap.ACCESS_READ)

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

active = {}
compute_nodes = 0
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
                    0x4C535633, 3, 0, 40, 40, 5120, 8, 512, 64, 64, 0x3F,
                ])
            elif opcode == -13:
                file_type = 14 if args.fake_mode == "identity-file-type" else 15
                model_sha256 = (
                    "f" * 64
                    if args.fake_mode == "identity-model-sha"
                    else args.model_sha256
                )
                send_i32(connection, [0x4C534944, 1, file_type])
                connection.sendall(bytes.fromhex(model_sha256))
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
                if any(item <= 0 for item in request_ids + route_epochs):
                    raise RuntimeError("nonpositive identity")
                for request_id, route_epoch, seq_id, position in zip(
                    request_ids, route_epochs, seq_ids, positions
                ):
                    current = active.get(seq_id)
                    if current is None:
                        if position != 0:
                            raise RuntimeError("position")
                        active[seq_id] = [request_id, route_epoch, 1]
                    else:
                        if (
                            current[0] != request_id
                            or current[1] != route_epoch
                            or current[2] != position
                        ):
                            raise RuntimeError("lineage")
                        current[2] += 1
                if args.fake_mode == "bad-lineage":
                    request_ids = (request_ids[0] + 1, *request_ids[1:])
                compute_nodes += count
                send_i32(connection, [0, count, 0])
                connection.sendall(struct.pack(f"<{count}q", *request_ids))
                connection.sendall(struct.pack(f"<{count}q", *route_epochs))
                connection.sendall(struct.pack(f"<{count}i", *seq_ids))
                connection.sendall(struct.pack(f"<{count}i", *positions))
                send_i32(connection, [token + 1 for token in tokens])
            elif opcode == -10:
                version, seq_id = recv_i32(connection, 2)
                request_id, route_epoch = struct.unpack(
                    "<2q", recv_exact(connection, 16)
                )
                current = active.pop(seq_id)
                if current[:2] != [request_id, route_epoch]:
                    raise RuntimeError("remove lineage")
                send_i32(connection, [0, version, len(active), 8, 0])
            elif opcode == -1:
                break
            else:
                raise RuntimeError(f"bad opcode {opcode}")

get_rows = 8
matmul = compute_nodes - get_rows
if args.fake_mode == "cpu-placement":
    buffers = {"CPU": compute_nodes}
    nested = {"GET_ROWS": {"CPU": get_rows}, "MUL_MAT": {"CPU": matmul}}
elif args.fake_mode == "host-gemm":
    buffers = {"CUDA0": compute_nodes - 2, "CUDA_Host": 2}
    nested = {
        "GET_ROWS": {"CUDA0": get_rows - 1, "CUDA_Host": 1},
        "MUL_MAT": {"CUDA0": matmul - 1, "CUDA_Host": 1},
    }
else:
    buffers = {"CUDA0": compute_nodes - 1, "CUDA_Host": 1}
    nested = {
        "GET_ROWS": {"CUDA0": get_rows - 1, "CUDA_Host": 1},
        "MUL_MAT": {"CUDA0": matmul},
    }
certificate = {
    "schema": "layersplit-scheduled-placement-v2",
    "role": "monov3",
    "mode": "monov3",
    "layer_start": 0,
    "layer_end": 40,
    "n_layer": 40,
    "pid": os.getpid(),
    "run_rc": 0,
    "compute_nodes": compute_nodes,
    "copy_nodes": 0,
    "metadata_nodes": 16,
    "missing_buffer_compute_nodes": 0,
    "compute_by_buffer_type": buffers,
    "compute_by_op": {"GET_ROWS": get_rows, "MUL_MAT": matmul},
    "compute_by_op_and_buffer": nested,
    "copy_by_buffer_type": {},
    "status": "SCHEDULED_PLACEMENT_OK",
}
print(
    "PLACEMENTCERT "
    + json.dumps(certificate, ensure_ascii=True, separators=(",", ":")),
    flush=True,
)
memory_certificate = {
    "schema": "layersplit-memory-breakdown-v1",
    "role": "wrong" if args.fake_mode == "memory-wrong-role" else "monov3",
    "pid": os.getpid() + (1 if args.fake_mode == "memory-wrong-pid" else 0),
    "model_buffer_bytes": 4096,
    "kv_buffer_bytes": 0 if args.fake_mode == "memory-zero" else 2048,
    "compute_buffer_bytes": 1024,
    "host_model_buffer_bytes": 128,
    "host_context_buffer_bytes": 64,
    "host_compute_buffer_bytes": 32,
}
if args.fake_mode != "memory-absent":
    print(
        "MEMORYCERT "
        + json.dumps(memory_certificate, ensure_ascii=True, separators=(",", ":")),
        flush=True,
    )
    if args.fake_mode == "memory-duplicate":
        print(
            "MEMORYCERT "
            + json.dumps(memory_certificate, ensure_ascii=True, separators=(",", ":")),
            flush=True,
        )
'''


class ProducerFixture:
    def __init__(self, test: unittest.TestCase, mode: str = "good"):
        self.temporary = tempfile.TemporaryDirectory()
        test.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        bundle = self.root / "bundle"
        bundle.mkdir()
        source = PRODUCER.read_text(encoding="ascii")
        if source.count(PRODUCTION_ROOTS_SOURCE) != 1:
            raise RuntimeError("production system-root source drift")
        if source.count(PRODUCTION_MAP_SOURCE) != 1:
            raise RuntimeError("production model-map source drift")
        self.producer = bundle / "cuda_monolithic_v1.py"
        self.producer.write_text(
            source.replace(
                PRODUCTION_ROOTS_SOURCE,
                TEST_ROOTS_SOURCE,
            ).replace(
                PRODUCTION_MAP_SOURCE,
                TEST_MAP_SOURCE,
            ),
            encoding="ascii",
        )
        self.pre = self.root / "pre"
        self.pre.mkdir()
        self.output = self.root / "result.json"
        self.port = free_port()
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"model-map-test" * 4096)

        corpus = [
            json.loads(line)
            for line in FROZEN_CORPUS.read_text(encoding="ascii").splitlines()
        ]
        wrapped = [
            {
                "acquisition_id": PHASE_ID,
                **row,
                "event_ns": 1,
                "kind": "corpus_item",
                "phase": "A_ONLY",
                "phase_id": PHASE_ID,
                "role": "quality.corpus",
            }
            for row in corpus
        ]
        corpus_raw = b"".join(canonical(row) for row in wrapped)
        phase_lock = {
            "artifact_root_sha256": "1" * 64,
            "candidate_sha256": "2" * 64,
            "contract_sha256": "3" * 64,
            "device_boot_ids": {
                "cuda": "11111111-1111-4111-8111-111111111111",
                "op12": "22222222-2222-4222-8222-222222222222",
                "op15": "33333333-3333-4333-8333-333333333333",
            },
            "event_ns": 1,
            "model_id": "qwen3-14b-q4_k_m",
            "phase": "A_ONLY",
            "phase_id": PHASE_ID,
            "preparation_sha256": "4" * 64,
            "quality_corpus_sha256": sha256(corpus_raw),
            "runtime_bundle_plan_sha256": "5" * 64,
            "schema": "s39-cp0-r1-phase-lock-v2.4",
        }
        (self.pre / "phase_lock.jsonl").write_bytes(canonical(phase_lock))
        (self.pre / "quality_corpus.jsonl").write_bytes(corpus_raw)

        length_pattern = [13, 9, 7, 11, 8, 12, 6, 10]
        lengths = [length_pattern[index % 8] for index in range(64)]
        histories = [
            [1000 + item_index * 100 + offset for offset in range(length)]
            for item_index, length in enumerate(lengths)
        ]
        requests = []
        for index in range(64):
            row = corpus[index]
            prompt = (
                f"Question: {row['question']}\n"
                f"A. {row['choices'][0]}\n"
                f"B. {row['choices'][1]}\n"
                f"C. {row['choices'][2]}\n"
                f"D. {row['choices'][3]}\n"
                "Answer with exactly one uppercase letter: A, B, C, or D.\n"
                "Answer:"
            ).encode("utf-8")
            requests.append({
                "item_index": index,
                "prompt_sha256": sha256(prompt),
                "prompt_utf8_base64": base64.b64encode(prompt).decode("ascii"),
                "prompt_utf8_bytes": len(prompt),
                "request_id": index % 8 + 1,
                "seq_id": index % 8,
                "token_ids": histories[index],
            })

        def make_group(group_index: int) -> dict:
            item_indices = list(range(group_index * 8, group_index * 8 + 8))
            waves = []
            for position in range(max(lengths[index] for index in item_indices)):
                waves.append([
                    {
                        "item_index": item_index,
                        "position": position,
                        "request_id": sequence + 1,
                        "seq_id": sequence,
                        "token_id": histories[item_index][position],
                    }
                    for sequence, item_index in enumerate(item_indices)
                    if position < lengths[item_index]
                ])
            partitions = []
            current = []
            for wave in waves:
                if current and len(current) + len(wave) > 64:
                    partitions.append({
                        "call_index": len(partitions),
                        "rows": current,
                    })
                    current = []
                current.extend(wave)
            if current:
                partitions.append({
                    "call_index": len(partitions),
                    "rows": current,
                })
            decode_calls = [
                {
                    "call_index": len(partitions) + ordinal,
                    "continuation_input_ordinal": ordinal,
                    "continuation_output_ordinal": ordinal + 1,
                    "rows": [
                        {
                            "item_index": item_index,
                            "position": lengths[item_index] + ordinal,
                            "request_id": sequence + 1,
                            "seq_id": sequence,
                        }
                        for sequence, item_index in enumerate(item_indices)
                    ],
                }
                for ordinal in range(7)
            ]
            return {
                "decode_calls": decode_calls,
                "group_index": group_index,
                "item_indices": item_indices,
                "prefill_partitions": partitions,
            }

        groups = [make_group(group_index) for group_index in range(8)]
        self.history = self.root / "history.json"
        self.history_value = {
            "batch": 8,
            "candidate_sha256": (
                "ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8"
            ),
            "continuation_tokens_per_request": 8,
            "corpus_sha256": (
                "3ffafee1615ae2de690a2726b880823e167a3d9c210c5faed86d8f0e93ecff4f"
            ),
            "mechanics_b8": groups[0],
            "model_id": "qwen3-14b-q4_k_m",
            "model_sha256": MODEL_SHA256,
            "n_batch": 64,
            "n_ctx_seq": 512,
            "n_ubatch": 64,
            "prefill_chunking": "WHOLE_POSITION_WAVES_MAX_64_ROWS",
            "prefill_row_order": "POSITION_MAJOR_THEN_ITEM_INDEX",
            "quality_groups": groups,
            "requests": requests,
            "schema": "s39-cp0-r1-token-history-v2.4",
            "tokenizer": {
                "component_id": "tokenizer",
                "path": "/opt/s39/tokenizer",
                "plan_sha256": "9" * 64,
                "sha256": "a" * 64,
            },
        }
        self.history.write_bytes(canonical(self.history_value))

        source_python = Path("/usr/bin/python3").resolve()
        self.launcher = bundle / "llama-layersplit"
        shutil.copyfile(source_python, self.launcher)
        self.launcher.chmod(0o755)
        self.extra_model = self.root / "other.gguf"
        self.extra_model.write_bytes(b"other-model" * 4096)
        command = [
            str(self.launcher),
            "-I",
            "-B",
            "-c",
            WORKER_SOURCE,
            "--port",
            str(self.port),
            "--model-sha256",
            MODEL_SHA256,
            "--fake-mode",
            mode,
            "--mode",
            "monov3",
            "-m",
            str(self.model),
            "--devices",
            "CUDA0",
            "--driver-batch",
            "8",
            "--driver-context",
            "512",
            "--driver-max-prefill",
            "8",
        ]
        if mode == "other-model":
            command.extend(["--extra-model", str(self.extra_model)])
        component = {
            "component_id": "llama-layersplit",
            "path": str(self.launcher),
            "sha256": sha256(self.launcher.read_bytes()),
            "stat": file_stat(self.launcher),
        }
        bundle_identity = {
            "bundle_id": "cuda_monolithic",
            "components": [component],
            "endpoint": "cuda",
            "launcher_component_id": "llama-layersplit",
            "process_role": "cuda_monolithic",
            "schema": "s39-cp0-r1-runtime-bundle-root-identity-v2.4",
        }
        self.launch = self.root / "launch.json"
        self.launch_value = {
            "allowed_system_roots": [
                "/mnt/storage/s21_deps/cuda-13.2.1/lib/",
                "/usr/lib/python3.12/",
                "/usr/lib/x86_64-linux-gnu/",
            ],
            "bundle_id": "cuda_monolithic",
            "bundle_root": str(bundle),
            "bundle_sha256": sha256(canonical(bundle_identity)),
            "command": command,
            "cwd": str(self.root),
            "endpoint": "cuda",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "HOME": "/home/zhihao",
                "LAYERSPLIT_MEMORY_CERT": "1",
                "LAYERSPLIT_MODEL_SHA256": MODEL_SHA256,
                "LAYERSPLIT_PLACEMENT_CERT": "1",
                "LC_ALL": "C",
                "LD_LIBRARY_PATH": str(bundle),
            },
            "expected_capabilities": 0x3F,
            "expected_file_type": 15,
            "expected_max_streams": 8,
            "expected_n_batch": 64,
            "expected_n_ctx_seq": 512,
            "expected_n_embd": 5120,
            "expected_n_layer": 40,
            "expected_n_ubatch": 64,
            "host": "127.0.0.1",
            "io_timeout_ms": 5000,
            "launcher_component_id": "llama-layersplit",
            "model_artifact": {
                "path": str(self.model),
                "sha256": MODEL_SHA256,
                "stat": file_stat(self.model),
            },
            "model_id": "qwen3-14b-q4_k_m",
            "model_sha256": MODEL_SHA256,
            "port": self.port,
            "required_components": [component],
            "route_epoch": 11,
            "schema": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
            "shutdown_timeout_ms": 5000,
            "startup_timeout_ms": 5000,
        }
        self.launch.write_bytes(canonical(self.launch_value))

    def rewrite_history(self) -> None:
        self.history.write_bytes(canonical(self.history_value))

    def rewrite_launch(self) -> None:
        self.launch.write_bytes(canonical(self.launch_value))

    def argv(self) -> list[str]:
        return [
            sys.executable,
            "-I",
            "-B",
            str(self.producer),
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
            MECHANISM_SHA256,
            "--model-sha256",
            MODEL_SHA256,
            "--histories",
            str(self.history),
            "--histories-sha256",
            sha256(self.history.read_bytes()),
            "--launch-plan",
            str(self.launch),
            "--launch-plan-sha256",
            sha256(self.launch.read_bytes()),
            "--execute",
            "--confirm",
            "RUN_V24_CUDA_MONOLITHIC_A_ONLY",
        ]

    def run(self) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            self.argv(),
            capture_output=True,
            check=False,
            timeout=30,
        )


class CudaMonolithicProducerTests(unittest.TestCase):
    def test_paid_input_digests_are_checked_before_parse(self):
        spec = importlib.util.spec_from_file_location(
            "cuda_monolithic_paid_input_digest_test",
            PRODUCER,
        )
        self.assertIsNotNone(spec)
        producer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(producer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, loader, error in (
                (
                    "histories.json",
                    lambda path, expected: producer.load_histories(
                        path,
                        "0" * 64,
                        "1" * 64,
                        [],
                        [],
                        expected,
                    ),
                    "E_HISTORIES_SHA256",
                ),
                (
                    "launch.json",
                    lambda path, expected: producer.load_launch(
                        path,
                        "0" * 64,
                        expected,
                    ),
                    "E_LAUNCH_PLAN_SHA256",
                ),
            ):
                with self.subTest(name=name):
                    path = root / name
                    original = producer.canonical_bytes({"original": True})
                    expected = producer.sha256(original)
                    path.write_bytes(
                        producer.canonical_bytes({"mutated": True})
                    )
                    with self.assertRaisesRegex(
                        producer.CaptureError,
                        error,
                    ):
                        loader(path, expected)

    def test_execution_confirmation_is_required(self):
        fixture = ProducerFixture(self)
        argv = fixture.argv()
        execute_index = argv.index("--execute")
        del argv[execute_index:execute_index + 3]
        process = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=30,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_EXECUTE_CONFIRMATION", process.stderr)

        fixture = ProducerFixture(self)
        argv = fixture.argv()
        argv[argv.index("--confirm") + 1] = "RUN_WRONG_PHASE"
        process = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=30,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_EXECUTE_CONFIRMATION", process.stderr)

    def test_production_system_roots_are_exact(self):
        spec = importlib.util.spec_from_file_location(
            "cuda_monolithic_v24_roots",
            PRODUCER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            module.ALLOWED_SYSTEM_ROOTS,
            [
                "/mnt/storage/s21_deps/cuda-13.2.1/lib/",
                "/usr/lib/x86_64-linux-gnu/",
            ],
        )
        self.assertEqual(
            module.EXPECTED_MODEL_MAP_OFFSETS,
            (0x26645000, 0x2188BD000),
        )
        self.assertEqual(module.EXPECTED_MODEL_MAP_PERMISSIONS, "r--s")

    def test_fake_native_protocol_emits_exact_rows_and_runtime_receipt(self):
        fixture = ProducerFixture(self)
        process = fixture.run()
        self.assertEqual(process.returncode, 0, process.stderr.decode())
        self.assertEqual(process.stdout, b"")
        result = json.loads(fixture.output.read_bytes())
        self.assertEqual(result["schema"], "s39-cp0-r1-v24-cuda-monolithic-raw-v1")
        rows = result["oracle_cuda_monolithic_rows"]
        self.assertEqual(len(rows), 9)
        expected_calls = (
            len(fixture.history_value["mechanics_b8"]["prefill_partitions"])
            + len(fixture.history_value["mechanics_b8"]["decode_calls"])
        )
        self.assertEqual(len(rows[0]["call_shapes"]), expected_calls)
        for index, row in enumerate(rows[1:]):
            self.assertEqual(row["request_id"], index + 1)
            self.assertEqual(len(row["continuation_tokens"]), 8)
            self.assertEqual(
                row["positions"],
                list(range(len(row["input_tokens"]))),
            )
        runtime = result["runtime_process"]
        producer_artifact = result["producer_artifact"]
        producer_receipt = result["producer_process_receipt"]
        self.assertEqual(result["producer_sha256"], producer_artifact["sha256"])
        self.assertEqual(producer_artifact["path"], str(fixture.producer))
        self.assertEqual(producer_receipt["argv"][0], str(fixture.producer))
        self.assertEqual(producer_receipt["source_path"], str(fixture.producer))
        self.assertEqual(
            producer_receipt["source_sha256"],
            producer_artifact["sha256"],
        )
        self.assertEqual(
            result["memory_certificate"],
            {
                "compute_buffer_bytes": 1024,
                "host_compute_buffer_bytes": 32,
                "host_context_buffer_bytes": 64,
                "host_model_buffer_bytes": 128,
                "kv_buffer_bytes": 2048,
                "model_buffer_bytes": 4096,
                "pid": runtime["pid"],
                "role": "monov3",
                "schema": "layersplit-memory-breakdown-v1",
            },
        )
        self.assertEqual(
            result["protocol_identity"],
            {
                "capabilities": 0x3F,
                "file_type": 15,
                "layer_end": 40,
                "layer_start": 0,
                "max_streams": 8,
                "model_sha256": MODEL_SHA256,
                "n_batch": 64,
                "n_ctx_seq": 512,
                "n_embd": 5120,
                "n_layer": 40,
                "n_ubatch": 64,
                "schema": "layersplit-stage-v3-identity-v1",
                "stage_identity_version": 1,
                "stage_protocol_version": 3,
            },
        )
        launch_binding = result["launch_binding"]
        self.assertEqual(launch_binding["launch"], fixture.launch_value)
        self.assertEqual(
            launch_binding["launch_plan_sha256"],
            sha256(fixture.launch.read_bytes()),
        )
        self.assertEqual(launch_binding["runtime_boot_id"], runtime["boot_id"])
        self.assertEqual(launch_binding["runtime_pid"], runtime["pid"])
        self.assertEqual(
            launch_binding["runtime_start_ticks"],
            runtime["start_ticks"],
        )
        self.assertEqual(runtime["pid"], result["runtime_model_binding"]["pid"])
        self.assertEqual(
            result["runtime_model_binding"]["other_gguf_mapping_paths"],
            [],
        )
        self.assertEqual(runtime["start_ticks"], result["runtime_model_binding"]["start_ticks"])
        self.assertEqual(runtime["loaded_repo_component_ids"], ["llama-layersplit"])
        self.assertTrue(result["runtime_model_binding"]["model_mapping_rows"])
        certificate_line = next(
            line
            for line in Path(result["worker_log_path"]).read_bytes().splitlines()
            if line.startswith(b"PLACEMENTCERT ")
        )
        self.assertTrue(
            certificate_line.startswith(
                b'PLACEMENTCERT {"schema":"layersplit-scheduled-placement-v2",'
                b'"role":"monov3","mode":"monov3","layer_start":0,'
            )
        )
        self.assertEqual(
            result["placement_certificate"]["compute_by_buffer_type"],
            {
                "CUDA0": sum(
                    len(row["input_tokens"]) + 7
                    for row in rows[1:]
                ) - 1,
                "CUDA_Host": 1,
            },
        )

    def test_producer_source_mutation_is_rejected(self):
        fixture = ProducerFixture(self)
        process = fixture.run()
        self.assertEqual(process.returncode, 0, process.stderr.decode())
        result = json.loads(fixture.output.read_bytes())
        fixture.producer.write_bytes(fixture.producer.read_bytes() + b"\n")
        spec = importlib.util.spec_from_file_location(
            "cuda_monolithic_v24_source_receipt",
            PRODUCER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaisesRegex(
            module.CaptureError,
            "E_PRODUCER_ARTIFACT_BYTES|E_PRODUCER_SOURCE_MUTATED|"
            "E_PRODUCER_SOURCE_STAT",
        ):
            module.validate_producer_identity(
                result["producer_artifact"],
                result["producer_process_receipt"],
            )

    def test_placement_json_duplicate_nonfinite_and_extra_keys_refuse(self):
        fixture = ProducerFixture(self)
        process = fixture.run()
        self.assertEqual(process.returncode, 0, process.stderr.decode())
        result = json.loads(fixture.output.read_bytes())
        spec = importlib.util.spec_from_file_location(
            "cuda_monolithic_v24_placement_json",
            PRODUCER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = result["placement_certificate"]
        runtime = result["runtime_process"]
        raw = json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        mutations = [
            raw[:-1] + b',"status":"SCHEDULED_PLACEMENT_OK"}',
            raw.replace(
                b'"metadata_nodes":16',
                b'"metadata_nodes":NaN',
                1,
            ),
            raw[:-1] + b',"unexpected":1}',
        ]
        for mutated in mutations:
            with self.subTest(mutated=mutated[-48:]):
                with self.assertRaises(module.CaptureError):
                    module.parse_placement_certificate(
                        b"PLACEMENTCERT " + mutated + b"\n",
                        runtime,
                    )

    def test_missing_or_wrong_model_sha_env_refuses_before_launch(self):
        fixture = ProducerFixture(self)
        fixture.launch_value["env"].pop("LAYERSPLIT_MODEL_SHA256")
        fixture.rewrite_launch()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_LAUNCH_MODEL_SHA256_ENV", process.stderr)

        fixture = ProducerFixture(self)
        fixture.launch_value["env"]["LAYERSPLIT_MODEL_SHA256"] = "f" * 64
        fixture.rewrite_launch()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(b"E_LAUNCH_MODEL_SHA256_ENV", process.stderr)

        fixture = ProducerFixture(self)
        fixture.launch_value["env"]["LD_PRELOAD"] = "/tmp/unbound.so"
        fixture.rewrite_launch()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_LAUNCH_ENV", process.stderr)

    def test_nonpositive_wire_identity_refuses_before_launch(self):
        fixture = ProducerFixture(self)
        fixture.history_value["requests"][0]["request_id"] = 0
        fixture.history_value["mechanics_b8"]["prefill_partitions"][0][
            "rows"
        ][0]["request_id"] = 0
        for call in fixture.history_value["mechanics_b8"]["decode_calls"]:
            call["rows"][0]["request_id"] = 0
        fixture.rewrite_history()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_HISTORY_REQUEST_IDENTITY", process.stderr)

    def test_split_position_wave_refuses_before_launch(self):
        fixture = ProducerFixture(self)
        first = fixture.history_value["mechanics_b8"]["prefill_partitions"][0]
        second = fixture.history_value["mechanics_b8"]["prefill_partitions"][1]
        moved = first["rows"].pop()
        second["rows"].insert(0, moved)
        fixture.rewrite_history()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_PREFILL_SPLIT_WAVE", process.stderr)

    def test_prompt_hash_mismatch_refuses_before_launch(self):
        fixture = ProducerFixture(self)
        fixture.history_value["requests"][0]["prompt_sha256"] = "0" * 64
        fixture.rewrite_history()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_HISTORY_PROMPT_SHA256", process.stderr)

    def test_all_quality_groups_are_load_bearing(self):
        fixture = ProducerFixture(self)
        row = fixture.history_value["quality_groups"][7]["prefill_partitions"][0][
            "rows"
        ][0]
        row["token_id"] += 1
        fixture.rewrite_history()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_PREFILL_TOKEN", process.stderr)

        fixture = ProducerFixture(self)
        fixture.history_value["quality_groups"][7]["decode_calls"][0][
            "rows"
        ].pop()
        fixture.rewrite_history()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_DECODE_ROWS", process.stderr)

    def test_mechanics_group_must_equal_quality_group_zero(self):
        fixture = ProducerFixture(self)
        fixture.history_value["mechanics_b8"] = copy.deepcopy(
            fixture.history_value["mechanics_b8"]
        )
        fixture.history_value["mechanics_b8"]["item_indices"] = list(range(1, 9))
        fixture.rewrite_history()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_HISTORY_MECHANICS_BINDING", process.stderr)

    def test_context_and_token_envelope_fail_closed(self):
        fixture = ProducerFixture(self)
        fixture.launch_value["expected_n_ctx_seq"] = 256
        context_index = fixture.launch_value["command"].index("--driver-context") + 1
        fixture.launch_value["command"][context_index] = "256"
        fixture.rewrite_launch()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_LAUNCH_EXPECTED_N_CTX_SEQ", process.stderr)

        fixture = ProducerFixture(self)
        fixture.history_value["requests"][63]["token_ids"] = list(range(505))
        fixture.rewrite_history()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_HISTORY_ROW", process.stderr)

        fixture = ProducerFixture(self)
        fixture.history_value["requests"][63]["token_ids"][0] = 151936
        fixture.rewrite_history()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_HISTORY_ROW", process.stderr)

    def test_lineage_mismatch_refuses(self):
        fixture = ProducerFixture(self, mode="bad-lineage")
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_BATCH_LINEAGE", process.stderr)

    def test_missing_and_extra_model_maps_refuse(self):
        fixture = ProducerFixture(self, mode="no-map")
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_RUNTIME_MODEL_MAP_COUNT", process.stderr)

        fixture = ProducerFixture(self, mode="other-model")
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_RUNTIME_OTHER_MODEL_MAP", process.stderr)

    def test_real_shaped_model_mapping_gate_is_exact(self):
        spec = importlib.util.spec_from_file_location(
            "cuda_monolithic_v24_model_maps",
            PRODUCER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_path = Path("/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf")
        device_id = os.makedev(103, 5)
        model_stat = {"device_id": device_id, "inode": 12863259}
        rows = [
            {
                "address_range": "70000000-70001000",
                "device_major": 103,
                "device_minor": 5,
                "inode": 12863259,
                "offset_bytes": 0x26645000,
                "path": str(model_path),
                "permissions": "r--s",
            },
            {
                "address_range": "71000000-71001000",
                "device_major": 103,
                "device_minor": 5,
                "inode": 12863259,
                "offset_bytes": 0x2188BD000,
                "path": str(model_path),
                "permissions": "r--s",
            },
        ]
        module.validate_model_mapping_rows(rows, model_path, model_stat)

        cases = []
        missing = copy.deepcopy(rows)
        missing.pop()
        cases.append((missing, "E_RUNTIME_MODEL_MAP_COUNT"))
        extra = copy.deepcopy(rows)
        extra.append({
            **copy.deepcopy(rows[1]),
            "address_range": "72000000-72001000",
            "offset_bytes": 1234,
        })
        cases.append((extra, "E_RUNTIME_MODEL_MAP_COUNT"))
        duplicate = copy.deepcopy(rows)
        duplicate[1]["offset_bytes"] = duplicate[0]["offset_bytes"]
        cases.append((duplicate, "E_RUNTIME_MODEL_MAP_DUPLICATE"))
        wrong_offset = copy.deepcopy(rows)
        wrong_offset[1]["offset_bytes"] += 4096
        cases.append((wrong_offset, "E_RUNTIME_MODEL_MAP_OFFSETS"))
        wrong_permissions = copy.deepcopy(rows)
        wrong_permissions[1]["permissions"] = "rw-s"
        cases.append((wrong_permissions, "E_RUNTIME_MODEL_MAP_IDENTITY"))
        for mutated, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(module.CaptureError, error):
                    module.validate_model_mapping_rows(
                        mutated,
                        model_path,
                        model_stat,
                    )

    def test_cpu_placement_refuses_after_raw_execution(self):
        fixture = ProducerFixture(self, mode="cpu-placement")
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_PLACEMENT_CERT_BACKEND", process.stderr)

    def test_cuda_host_non_get_rows_refuses_with_consistent_totals(self):
        fixture = ProducerFixture(self, mode="host-gemm")
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_PLACEMENT_CERT_HOST_OP", process.stderr)

    def test_memory_certificate_failures_are_fatal(self):
        cases = [
            ("memory-absent", b"E_MEMORY_CERT_COUNT"),
            ("memory-duplicate", b"E_MEMORY_CERT_COUNT"),
            ("memory-zero", b"E_MEMORY_CERT_DEVICE_BYTES"),
            ("memory-wrong-role", b"E_MEMORY_CERT_IDENTITY"),
            ("memory-wrong-pid", b"E_MEMORY_CERT_IDENTITY"),
        ]
        for mode, error in cases:
            with self.subTest(mode=mode):
                fixture = ProducerFixture(self, mode=mode)
                process = fixture.run()
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse(fixture.output.exists())
                self.assertIn(error, process.stderr)

    def test_observed_protocol_identity_failures_are_fatal(self):
        for mode in ("identity-file-type", "identity-model-sha"):
            with self.subTest(mode=mode):
                fixture = ProducerFixture(self, mode=mode)
                process = fixture.run()
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse(fixture.output.exists())
                self.assertIn(b"E_MODEL_IDENTITY", process.stderr)

    def test_component_stat_mutation_refuses_before_launch(self):
        fixture = ProducerFixture(self)
        os.utime(fixture.launcher, ns=(1, 1))
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_LAUNCH_COMPONENT_STAT", process.stderr)

    def test_mode_and_existing_output_fail_closed(self):
        fixture = ProducerFixture(self)
        mode_index = fixture.launch_value["command"].index("--mode") + 1
        fixture.launch_value["command"][mode_index] = "stagenet"
        fixture.rewrite_launch()
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_LAUNCH_OPTION_VALUE", process.stderr)

        fixture = ProducerFixture(self)
        fixture.output.write_bytes(b"occupied")
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(fixture.output.read_bytes(), b"occupied")
        self.assertIn(b"E_OUTPUT", process.stderr)

    def test_runtime_recheck_rejects_pid_start_substitution(self):
        spec = importlib.util.spec_from_file_location("cuda_monolithic_v24", PRODUCER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = {"observed_ns": 1, "pid": 10, "start_ticks": 20}
        after = copy.deepcopy(before)
        after["observed_ns"] = 2
        after["start_ticks"] += 1
        with self.assertRaisesRegex(module.CaptureError, "E_RUNTIME_RECHECK"):
            module.require_same_process(before, after)

    def test_noncanonical_history_refuses(self):
        fixture = ProducerFixture(self)
        fixture.history.write_text(
            '{"schema":"s39-cp0-r1-token-history-v2.4",'
            '"schema":"duplicate"}\n',
            encoding="ascii",
        )
        process = fixture.run()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(fixture.output.exists())
        self.assertIn(b"E_DUPLICATE_KEY", process.stderr)


if __name__ == "__main__":
    unittest.main()
