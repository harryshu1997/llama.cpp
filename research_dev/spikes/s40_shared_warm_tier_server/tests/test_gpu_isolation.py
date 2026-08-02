#!/usr/bin/env python3

import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
S40 = HERE.parent
sys.path.insert(0, str(S40))

from evidence_common import (  # noqa: E402
    EvidenceError,
    canonical_bytes,
    digest_file,
)
from gpu_isolation import (  # noqa: E402
    ExclusiveGpuLock,
    _write_all,
    capture_sample,
    observation_commands,
    parse_cmdline,
    parse_gpu_identity,
    parse_gpu_processes,
    validate_lock_record,
    validate_observation,
)


GPU_UUID = "GPU-test"
GPU_NAME = "NVIDIA GeForce RTX 4060 Ti"
BOOT_ID = "boot-test"
RUN_ID = "run-test"


def encoded(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def process_stat(pid: int, start_ticks: int) -> bytes:
    fields = [
        "S", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "11", "12", "13", "14", "15", "16", "17", "18",
        str(start_ticks),
    ]
    return f"{pid} (llama server) {' '.join(fields)}\n".encode("ascii")


def sample(sequence: int, started: int, completed: int, busy: bool) -> dict:
    process_stdout = (
        b'123,GPU-test,"/tmp/llama-server",100\n' if busy else b"")
    observations = []
    if busy:
        observations.append({
            "cmdline_base64": encoded(b"/tmp/llama-server\0--flag\0"),
            "pid": 123,
            "stat_base64": encoded(process_stat(123, 900)),
        })
    return {
        "completed_ns": completed,
        "identity_stderr_base64": encoded(b""),
        "identity_stdout_base64": encoded(
            b"GPU-test,NVIDIA GeForce RTX 4060 Ti,00000000:01:00.0,0\n"),
        "process_observations": observations,
        "process_stderr_base64": encoded(b""),
        "process_stdout_base64": encoded(process_stdout),
        "sequence": sequence,
        "started_ns": started,
        "type": "SAMPLE",
    }


def fixture_rows() -> list[dict]:
    executable = Path("/usr/bin/nvidia-smi")
    identity, processes = observation_commands(executable, GPU_UUID)
    return [
        {
            "gpu_name": GPU_NAME,
            "gpu_uuid": GPU_UUID,
            "host_boot_id": BOOT_ID,
            "identity_argv": identity,
            "nvidia_smi_bytes": 100,
            "nvidia_smi_path": str(executable),
            "nvidia_smi_sha256": "1" * 64,
            "process_argv": processes,
            "run_id": RUN_ID,
            "schema": "s40-selected-gpu-observer-v1",
            "started_ns": 50,
            "type": "START",
        },
        sample(0, 100, 110, False),
        sample(1, 200, 210, True),
        sample(2, 300, 310, False),
        {
            "completed_ns": 320,
            "sample_count": 3,
            "type": "STOP",
        },
    ]


class GpuIsolationTests(unittest.TestCase):
    def write_rows(self, root: Path, rows: list[dict]) -> Path:
        path = root / "gpu-observer.jsonl"
        path.write_bytes(b"".join(canonical_bytes(row) for row in rows))
        return path

    def validate(self, path: Path) -> dict:
        return validate_observation(
            path,
            expected_run_id=RUN_ID,
            expected_uuid=GPU_UUID,
            expected_name=GPU_NAME,
            expected_boot_id=BOOT_ID,
            trace_start_ns=150,
            trace_end_ns=250,
            allowed_processes={
                (123, 900): ["/tmp/llama-server", "--flag"],
            },
            max_gap_ns=100,
        )

    def test_parse_csv_handles_quoted_process_name(self):
        rows = parse_gpu_processes(
            b'12,GPU-test,"a, process",34\n',
            GPU_UUID,
        )
        self.assertEqual(rows[0]["process_name"], "a, process")

    def test_identity_is_exact(self):
        row = parse_gpu_identity(
            b"GPU-test,NVIDIA GeForce RTX 4060 Ti,01:00.0,0\n",
            GPU_UUID,
            GPU_NAME,
        )
        self.assertEqual(row["gpu_index"], 0)
        with self.assertRaisesRegex(EvidenceError, "name mismatch"):
            parse_gpu_identity(
                b"GPU-test,other,01:00.0,0\n",
                GPU_UUID,
                GPU_NAME,
            )

    def test_cmdline_requires_complete_nul_framing(self):
        self.assertEqual(parse_cmdline(b"a\0b\0"), ["a", "b"])
        with self.assertRaisesRegex(EvidenceError, "incomplete"):
            parse_cmdline(b"a\0b")

    def test_observation_passes_with_bracket_and_exact_process(self):
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            path = self.write_rows(Path(directory), fixture_rows())
            result = self.validate(path)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["sample_count"], 3)

    def test_foreign_process_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            path = self.write_rows(Path(directory), fixture_rows())
            with self.assertRaisesRegex(EvidenceError, "foreign"):
                validate_observation(
                    path,
                    expected_run_id=RUN_ID,
                    expected_uuid=GPU_UUID,
                    expected_name=GPU_NAME,
                    expected_boot_id=BOOT_ID,
                    trace_start_ns=150,
                    trace_end_ns=250,
                    allowed_processes={},
                    max_gap_ns=100,
                )

    def test_pid_reuse_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            path = self.write_rows(Path(directory), fixture_rows())
            with self.assertRaisesRegex(EvidenceError, "foreign"):
                validate_observation(
                    path,
                    expected_run_id=RUN_ID,
                    expected_uuid=GPU_UUID,
                    expected_name=GPU_NAME,
                    expected_boot_id=BOOT_ID,
                    trace_start_ns=150,
                    trace_end_ns=250,
                    allowed_processes={
                        (123, 901): ["/tmp/llama-server", "--flag"],
                    },
                    max_gap_ns=100,
                )

    def test_command_mutation_is_rejected(self):
        rows = fixture_rows()
        rows[0]["identity_argv"][-1] = "--format=csv"
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            path = self.write_rows(Path(directory), rows)
            with self.assertRaisesRegex(EvidenceError, "command mismatch"):
                self.validate(path)

    def test_raw_process_mutation_is_rejected(self):
        rows = fixture_rows()
        rows[2]["process_stdout_base64"] = encoded(
            b"124,GPU-test,/tmp/llama-server,100\n")
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            path = self.write_rows(Path(directory), rows)
            with self.assertRaisesRegex(
                    EvidenceError, "process query mismatch"):
                self.validate(path)

    def test_missing_pre_or_post_idle_is_rejected(self):
        rows = fixture_rows()
        rows[1] = sample(0, 100, 110, True)
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            path = self.write_rows(Path(directory), rows)
            with self.assertRaisesRegex(EvidenceError, "before launch"):
                self.validate(path)
        rows = fixture_rows()
        rows[3] = sample(2, 300, 310, True)
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            path = self.write_rows(Path(directory), rows)
            with self.assertRaisesRegex(EvidenceError, "after cleanup"):
                self.validate(path)

    def test_trace_must_be_bracketed_and_gaps_bounded(self):
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            path = self.write_rows(Path(directory), fixture_rows())
            with self.assertRaisesRegex(EvidenceError, "not bracketed"):
                validate_observation(
                    path,
                    expected_run_id=RUN_ID,
                    expected_uuid=GPU_UUID,
                    expected_name=GPU_NAME,
                    expected_boot_id=BOOT_ID,
                    trace_start_ns=90,
                    trace_end_ns=250,
                    allowed_processes={
                        (123, 900): ["/tmp/llama-server", "--flag"],
                    },
                    max_gap_ns=100,
                )
            with self.assertRaisesRegex(EvidenceError, "gap exceeded"):
                validate_observation(
                    path,
                    expected_run_id=RUN_ID,
                    expected_uuid=GPU_UUID,
                    expected_name=GPU_NAME,
                    expected_boot_id=BOOT_ID,
                    trace_start_ns=150,
                    trace_end_ns=250,
                    allowed_processes={
                        (123, 900): ["/tmp/llama-server", "--flag"],
                    },
                    max_gap_ns=50,
                )

    def test_probe_duration_is_bounded(self):
        rows = fixture_rows()
        rows[2]["completed_ns"] = rows[2]["started_ns"] + 101
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            path = self.write_rows(Path(directory), rows)
            with self.assertRaisesRegex(
                    EvidenceError, "probe duration exceeded"):
                validate_observation(
                    path,
                    expected_run_id=RUN_ID,
                    expected_uuid=GPU_UUID,
                    expected_name=GPU_NAME,
                    expected_boot_id=BOOT_ID,
                    trace_start_ns=150,
                    trace_end_ns=250,
                    allowed_processes={
                        (123, 900): ["/tmp/llama-server", "--flag"],
                    },
                    max_gap_ns=1_000,
                    max_probe_duration_ns=100,
                )

    def test_nvidia_smi_mutation_during_sample_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            executable = Path(directory) / "nvidia-smi"
            executable.write_bytes(b"original\n")
            expected = digest_file(executable)
            identity_argv, process_argv = observation_commands(
                executable, GPU_UUID)
            calls = 0

            def runner(argv, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    executable.write_bytes(b"mutated\n")
                    stdout = (
                        b"GPU-test,NVIDIA GeForce RTX 4060 Ti,"
                        b"00000000:01:00.0,0\n"
                    )
                else:
                    stdout = b""
                return __import__("subprocess").CompletedProcess(
                    argv, 0, stdout=stdout, stderr=b"")

            with self.assertRaisesRegex(
                    EvidenceError, "changed after query"):
                capture_sample(
                    0,
                    identity_argv,
                    process_argv,
                    GPU_UUID,
                    GPU_NAME,
                    executable,
                    expected,
                    runner=runner,
                )

    def test_write_all_retries_short_writes(self):
        sink = bytearray()

        def short_write(_descriptor, raw):
            count = max(1, min(2, len(raw)))
            sink.extend(raw[:count])
            return count

        with patch("gpu_isolation.os.write", side_effect=short_write):
            _write_all(99, b"abcdef")
        self.assertEqual(bytes(sink), b"abcdef")

    def test_lock_is_exclusive_and_record_is_typed(self):
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            path = Path(directory) / "gpu.lock"
            first = ExclusiveGpuLock(path, GPU_UUID, RUN_ID, BOOT_ID)
            first.acquire()
            second = ExclusiveGpuLock(path, GPU_UUID, "other", BOOT_ID)
            with self.assertRaises(BlockingIOError):
                second.acquire()
            record = first.release()
            result = validate_lock_record(
                record, GPU_UUID, RUN_ID, BOOT_ID)
            self.assertGreaterEqual(
                result["released_ns"], result["acquired_ns"])

    def test_lock_symlink_is_rejected(self):
        if not hasattr(os, "O_NOFOLLOW"):
            self.skipTest("O_NOFOLLOW unavailable")
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"")
            link = root / "gpu.lock"
            link.symlink_to(target)
            lock = ExclusiveGpuLock(link, GPU_UUID, RUN_ID, BOOT_ID)
            with self.assertRaises(OSError):
                lock.acquire()

    def test_bool_is_not_accepted_as_numeric_evidence(self):
        record = {
            "acquired_ns": True,
            "device": 1,
            "gpu_uuid": GPU_UUID,
            "host_boot_id": BOOT_ID,
            "inode": 1,
            "lock_path": "/tmp/lock",
            "owner_pid": 1,
            "owner_start_ticks": 1,
            "released_ns": 2,
            "run_id": RUN_ID,
            "schema": "s40-selected-gpu-lock-v1",
        }
        with self.assertRaisesRegex(EvidenceError, "integer"):
            validate_lock_record(record, GPU_UUID, RUN_ID, BOOT_ID)

    def test_duplicate_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_gpu_") as directory:
            path = Path(directory) / "gpu.jsonl"
            path.write_bytes(
                b'{"type":"START","type":"START"}\n'
                b'{}\n{}\n{}\n')
            with self.assertRaisesRegex(EvidenceError, "duplicate JSON key"):
                validate_observation(
                    path,
                    expected_run_id=RUN_ID,
                    expected_uuid=GPU_UUID,
                    expected_name=GPU_NAME,
                    expected_boot_id=BOOT_ID,
                    trace_start_ns=1,
                    trace_end_ns=2,
                    allowed_processes={},
                )


if __name__ == "__main__":
    unittest.main()
