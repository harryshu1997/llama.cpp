#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evidence = load_module("v24_mono_evidence", HERE / "cp0_r1_evidence_v24.py")
common = evidence.common
base_tests = load_module("v24_mono_base", HERE / "tests" / "test_v24_readiness.py")
joint_tests = load_module(
    "v24_mono_joint",
    HERE / "tests" / "test_v24_joint_authority.py",
)
producer_tests = load_module(
    "v24_mono_producer_tests",
    HERE / "producers_v1" / "tests" / "test_joint_phone_cuda_v1.py",
)


def stat_record(path: Path) -> dict[str, int]:
    value = path.stat(follow_symlinks=False)
    return {
        "ctime_ns": value.st_ctime_ns,
        "device_id": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "mtime_ns": value.st_mtime_ns,
        "size": value.st_size,
    }


class MonolithicAuthorityFixture:
    def __init__(self, root: Path):
        self.root = root
        base_root = root / "base"
        base_root.mkdir()
        self.base = base_tests.Fixture(base_root)
        self.contract = self.base.contract
        self.candidate = self.base.candidate
        self.history = self.base.history
        self.history_raw = self.base.history_raw
        self.plan = copy.deepcopy(self.base.plan)
        self.runtime = self.base.runtime
        self.bundle = root / "bundle"
        self.bundle.mkdir()
        self.pre_dir = root / "pre"
        self.pre_dir.mkdir()
        source = HERE / "producers_v1" / "cuda_monolithic_v1.py"
        self.source = self.bundle / "cuda_monolithic_v1.py"
        self.source.write_bytes(source.read_bytes())
        source_pin = self.contract["producer_requirements"]["source_programs"][
            "cuda_monolithic"
        ]
        assert len(self.source.read_bytes()) == source_pin["bytes"]
        assert common.sha256_file(self.source) == source_pin["sha256"]
        self.plan["bundle_roots"]["cuda_monolithic"] = str(self.bundle)
        component = next(
            value for value in self.plan["components"]
            if value["component_id"] == "cuda-mono-capture"
        )
        component["path"] = str(self.source)
        component["bytes"] = source_pin["bytes"]
        component["sha256"] = source_pin["sha256"]
        self.launch_path = root / "cuda-launch.json"
        self.launch_path.write_bytes(
            common.canonical_bytes(self.plan["cuda_monolithic_launch"])
        )
        self.output = self.bundle / "cuda-monolithic.json"
        self.worker_log = Path(str(self.output) + ".worker.log")
        self.worker_log.write_bytes(b"worker\n")
        self.receipt = self._receipt()
        self.rows_by_role = self._rows_by_role()

    def _runtime_process(self) -> dict:
        return copy.deepcopy(next(
            value for value in self.runtime["processes"]
            if value["bundle_id"] == "cuda_monolithic"
        ))

    def _placement(self, pid: int) -> dict:
        return {
            "compute_by_buffer_type": {"CUDA0": 10},
            "compute_by_op": {"MUL_MAT": 10},
            "compute_by_op_and_buffer": {"MUL_MAT": {"CUDA0": 10}},
            "compute_nodes": 10,
            "copy_by_buffer_type": {},
            "copy_nodes": 0,
            "layer_end": 40,
            "layer_start": 0,
            "metadata_nodes": 0,
            "missing_buffer_compute_nodes": 0,
            "mode": "monov3",
            "n_layer": 40,
            "pid": pid,
            "role": "monov3",
            "run_rc": 0,
            "schema": "layersplit-scheduled-placement-v2",
            "status": "SCHEDULED_PLACEMENT_OK",
        }

    def _receipt(self) -> dict:
        model = next(
            value for value in self.candidate["models"]
            if value["slot"] == "A"
        )
        runtime_process = self._runtime_process()
        groups = joint_tests.execution_groups(self.history)
        rows = producer_tests.mechanics_rows(
            self.history,
            groups,
            "CUDA0",
        )
        for index, row in enumerate(rows):
            row["event_ns"] = 1500 + index
        producer_pid = runtime_process["pid"] + 1000
        producer_start = runtime_process["start_ticks"] - 1
        mechanism = "6" * 64
        argv = [
            str(self.source),
            "--output",
            str(self.output),
            "--phase-id",
            self.runtime["phase_id"],
            "--pre-dir",
            str(self.pre_dir),
            "--started",
            "1000",
            "--plan",
            common.sha256_bytes(common.canonical_bytes(self.plan)),
            "--mechanism-commands-sha256",
            mechanism,
            "--model-sha256",
            model["artifact"]["sha256"],
            "--histories",
            self.plan["token_history"]["artifact_path"],
            "--launch-plan",
            str(self.launch_path),
        ]
        source_raw = self.source.read_bytes()
        live_mapping = runtime_process["model_mapping"]
        placement = self._placement(runtime_process["pid"])
        return {
            "completed_ns": 1800,
            "history_binding": {
                "corpus_item_indices": list(range(8)),
                "histories_sha256": common.sha256_bytes(self.history_raw),
                "prompt_sha256s": [
                    self.history["requests"][index]["prompt_sha256"]
                    for index in range(8)
                ],
                "quality_corpus_sha256": self.contract["quality_corpus"][
                    "sha256"
                ],
                "token_history_corpus_sha256": self.history["corpus_sha256"],
            },
            "launch_binding": {
                "launch": self.plan["cuda_monolithic_launch"],
                "launch_plan_sha256": common.sha256_file(self.launch_path),
                "runtime_boot_id": runtime_process["boot_id"],
                "runtime_pid": runtime_process["pid"],
                "runtime_start_ticks": runtime_process["start_ticks"],
            },
            "mechanism_commands_sha256": mechanism,
            "memory_certificate": {
                "compute_buffer_bytes": 100,
                "host_compute_buffer_bytes": 20,
                "host_context_buffer_bytes": 30,
                "host_model_buffer_bytes": 40,
                "kv_buffer_bytes": 1_000_000_000,
                "model_buffer_bytes": 6_000_000_000,
                "pid": runtime_process["pid"],
                "role": "monov3",
                "schema": "layersplit-memory-breakdown-v1",
            },
            "model_id": evidence.MODEL_ID,
            "model_sha256": model["artifact"]["sha256"],
            "oracle_cuda_monolithic_rows": rows,
            "phase_id": self.runtime["phase_id"],
            "placement_certificate": placement,
            "producer_artifact": {
                "bytes": len(source_raw),
                "path": str(self.source),
                "sha256": common.sha256_bytes(source_raw),
                "stat": stat_record(self.source),
            },
            "producer_process_receipt": {
                "argv": argv,
                "boot_id": runtime_process["boot_id"],
                "cwd": str(self.bundle),
                "pid": producer_pid,
                "schema": "s39-cp0-r1-v24-producer-process-receipt-v1",
                "source_path": str(self.source),
                "source_sha256": common.sha256_bytes(source_raw),
                "start_ticks": producer_start,
            },
            "producer_sha256": common.sha256_bytes(source_raw),
            "protocol_identity": {
                "capabilities": 0x3F,
                "file_type": 15,
                "layer_end": 40,
                "layer_start": 0,
                "max_streams": 8,
                "model_sha256": model["artifact"]["sha256"],
                "n_batch": 64,
                "n_ctx_seq": 512,
                "n_embd": 5120,
                "n_layer": 40,
                "n_ubatch": 64,
                "schema": "layersplit-stage-v3-identity-v1",
                "stage_identity_version": 1,
                "stage_protocol_version": 3,
            },
            "runtime_model_binding": {
                "argv": live_mapping["argv"],
                "model_mapping_rows": live_mapping["model_mapping_rows"],
                "model_path": live_mapping["model_path"],
                "model_sha256": live_mapping["model_sha256"],
                "model_stat": live_mapping["pre_stat"],
                "other_gguf_mapping_paths": [],
                "pid": runtime_process["pid"],
                "start_ticks": runtime_process["start_ticks"],
            },
            "runtime_process": runtime_process,
            "schema": "s39-cp0-r1-v24-cuda-monolithic-raw-v1",
            "started_ns": 1300,
            "worker_log_bytes": len(self.worker_log.read_bytes()),
            "worker_log_path": str(self.worker_log),
            "worker_log_sha256": common.sha256_file(self.worker_log),
        }

    def _rows_by_role(self) -> dict[str, list[dict]]:
        role = f"model.{evidence.MODEL_ID}.oracle.cuda_monolithic"
        return {
            role: [
                {
                    "acquisition_id": self.runtime["phase_id"],
                    **row,
                    "phase": "A_ONLY",
                    "phase_id": self.runtime["phase_id"],
                    "role": role,
                }
                for row in self.receipt["oracle_cuda_monolithic_rows"]
            ]
        }

    def validate(self) -> None:
        evidence.validate_cuda_monolithic_receipt(
            self.receipt,
            self.bundle,
            self.contract,
            self.candidate,
            self.history,
            self.history_raw,
            self.plan,
            self.runtime,
            self.rows_by_role,
        )


class MonolithicAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = MonolithicAuthorityFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_producer_shaped_monolithic_receipt_passes(self):
        self.fixture.validate()

    def test_producer_source_bytes_are_load_bearing(self):
        self.fixture.source.write_bytes(self.fixture.source.read_bytes() + b"\n")
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_CUDA_PRODUCER_SOURCE_(BYTES|SHA256|STAT)",
        ):
            self.fixture.validate()

    def test_producer_source_stat_is_load_bearing(self):
        self.fixture.receipt["producer_artifact"]["stat"]["mtime_ns"] += 1
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_CUDA_PRODUCER_SOURCE_STAT",
        ):
            self.fixture.validate()

    def test_producer_pid_is_load_bearing(self):
        self.fixture.receipt["producer_process_receipt"]["pid"] = (
            self.fixture.receipt["runtime_process"]["pid"]
        )
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_CUDA_PRODUCER_PROCESS_PID_REUSE",
        ):
            self.fixture.validate()

    def test_producer_start_ticks_are_load_bearing(self):
        self.fixture.receipt["producer_process_receipt"]["start_ticks"] = (
            self.fixture.receipt["runtime_process"]["start_ticks"] + 1
        )
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_CUDA_PRODUCER_PROCESS_START",
        ):
            self.fixture.validate()

    def test_producer_boot_is_load_bearing(self):
        self.fixture.receipt["producer_process_receipt"]["boot_id"] = (
            "ffffffff-ffff-ffff-ffff-ffffffffffff"
        )
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_CUDA_PRODUCER_PROCESS_BOOT",
        ):
            self.fixture.validate()

    def test_producer_argv_is_load_bearing(self):
        argv = self.fixture.receipt["producer_process_receipt"]["argv"]
        argv[argv.index("--phase-id") + 1] = "wrong"
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_CUDA_PRODUCER_ARG_PHASE",
        ):
            self.fixture.validate()

    def test_producer_cwd_is_load_bearing(self):
        self.fixture.receipt["producer_process_receipt"]["cwd"] = str(
            self.fixture.root
        )
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_CUDA_PRODUCER_PROCESS_CWD",
        ):
            self.fixture.validate()


if __name__ == "__main__":
    unittest.main()
