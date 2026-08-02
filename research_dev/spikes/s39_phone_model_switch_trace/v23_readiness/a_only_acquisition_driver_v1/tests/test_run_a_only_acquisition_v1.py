#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
DRIVER_DIR = HERE.parent
V23 = DRIVER_DIR.parent
S39 = V23.parent
for path in (DRIVER_DIR, V23, S39, S39 / "tests"):
    sys.path.insert(0, str(path))

import acquisition_support_v1 as driver
from test_cp0_r1_evidence_v22 import V22PhaseFixture


def marker(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class ScriptedClock:
    def __init__(self, values: list[int]):
        self.values = iter(values)

    def __call__(self) -> int:
        return next(self.values)


class FakeRunner:
    def __init__(self, fragments: list[dict]):
        self.fragments = iter(fragments)
        self.calls: list[list[str]] = []

    def run(self, argv: list[str], *, timeout: int):
        del timeout
        self.calls.append(list(argv))
        output = Path(argv[argv.index("--output") + 1])
        output.write_bytes(driver.common.canonical_bytes(next(self.fragments)))
        return subprocess.CompletedProcess(argv, 0, b"", b"")


class AcquisitionFixture:
    def __init__(self, test: unittest.TestCase):
        self.temporary = tempfile.TemporaryDirectory()
        test.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (
            self.contract,
            self.contract_raw,
            self.candidate,
            self.candidate_raw,
            self.parent,
            self.corpus,
        ) = driver.v22.validate_inputs(
            driver.v22.DEFAULT_CONTRACT,
            driver.v23.DEFAULT_CANDIDATE,
        )
        self.phase = V22PhaseFixture(
            self.root / "fixture",
            "A_ONLY",
            self.contract,
            self.contract_raw,
            self.parent,
            self.candidate,
            self.candidate_raw,
            [],
            [],
            frozen_corpus=self.corpus,
        )
        self.phase_id = "cp0-r1-v23-a-only-driver-test"
        self.phase.set_phase_id(self.phase_id)
        self.phase.repair_model()
        self.model = self.candidate["models"][0]
        self.route = self.phase.lock
        self.start = self.phase.started_ns
        self.joint_complete = self.start + 900_000
        self.monolithic_start = self.joint_complete + 100
        self.monolithic_complete = self.monolithic_start + 100
        self.acquisition_started = self.start - 100
        self.pre_dir = self.root / "run" / "pre"
        self.artifact_dir = self.root / "run" / "artifact"
        self.fresh_dir = self.root / "run" / "fresh"
        self.output_dir = self.root / "run" / "acquisition"
        for path in (
            self.pre_dir,
            self.artifact_dir,
            self.fresh_dir,
            self.output_dir,
        ):
            path.mkdir(parents=True)
        self._write_pre()
        self.stats = self._stats()
        self.paths = {
            "cuda": self.route["cuda_model_path"],
            "op15": self.route["op15_shard_path"],
            "op12": self.route["op12_shard_path"],
            "op15_worker": "/data/local/tmp/s39-v23/llama-layersplit",
            "op12_worker": "/data/local/tmp/s39-v23/llama-layersplit",
        }
        self.worker_sha = {
            "op15_worker": marker("op15-worker"),
            "op12_worker": marker("op12-worker"),
        }
        self.artifact = self._artifact()
        self.artifact_raw = driver.common.canonical_bytes(self.artifact)
        (self.artifact_dir / "artifact_snapshot.json").write_bytes(
            self.artifact_raw
        )
        self.readiness = self._readiness_lock()
        self.readiness_raw = driver.common.canonical_bytes(self.readiness)
        (self.fresh_dir / "readiness_lock.json").write_bytes(
            self.readiness_raw
        )
        self.fresh = self._fresh()
        (self.fresh_dir / "fresh_snapshot.json").write_bytes(
            driver.common.canonical_bytes(self.fresh)
        )
        self.plan = self._plan()
        self.plan_path = self.root / "A_ONLY_COMMAND_PLAN_V1.json"
        self.plan_path.write_bytes(driver.common.canonical_bytes(self.plan))
        self.fragments = self._fragments()

    def _write_pre(self) -> None:
        mapping = {
            "phase.lock": "phase_lock.jsonl",
            "quality.corpus": "quality_corpus.jsonl",
            f"model.{driver.MODEL_ID}.route_lock": "route_lock.jsonl",
        }
        for role, filename in mapping.items():
            raw = b"".join(
                driver.v2.canonical_line(row)
                for row in self.phase.rows[role]
            )
            (self.pre_dir / filename).write_bytes(raw)

    def _stats(self) -> dict[str, dict[str, int]]:
        endpoints = ("cuda", "op15", "op12", "op15_worker", "op12_worker")
        result = {}
        for index, endpoint in enumerate(endpoints):
            if endpoint == "cuda":
                size = self.model["artifact"]["bytes"]
            elif endpoint.endswith("_worker"):
                size = 50_000_000 + index
            else:
                size = self.route[f"{endpoint}_shard_bytes"]
            result[endpoint] = {
                "ctime_ns": 100_000_000_000 + index,
                "device_id": index + 1,
                "inode": 1000 + index,
                "mode": 33188,
                "mtime_ns": 90_000_000_000 + index,
                "size": size,
            }
        return result

    def _artifact(self) -> dict:
        records = []
        paths = {
            "cuda": self.route["cuda_model_path"],
            "op15": self.route["op15_shard_path"],
            "op12": self.route["op12_shard_path"],
            "op15_worker": "/data/local/tmp/s39-v23/llama-layersplit",
            "op12_worker": "/data/local/tmp/s39-v23/llama-layersplit",
        }
        for endpoint in ("cuda", "op15", "op12", "op15_worker", "op12_worker"):
            if endpoint == "cuda":
                digest = self.model["artifact"]["sha256"]
            elif endpoint.endswith("_worker"):
                digest = self.worker_sha[endpoint]
            else:
                digest = self.route[f"{endpoint}_shard_sha256"]
            records.append({
                "bytes": self.stats[endpoint]["size"],
                "endpoint": endpoint,
                "path": paths[endpoint],
                "sha256": digest,
                "stat": copy.deepcopy(self.stats[endpoint]),
            })
        return {
            "artifacts": records,
            "completed_ns": self.acquisition_started - 1000,
            "model_id": driver.MODEL_ID,
            "phase": "A_ONLY",
            "route_lock_sha256": marker("route-lock"),
            "schema": "s39-cp0-r1-artifact-snapshot-v2.3",
            "slot": "A",
            "started_ns": self.acquisition_started - 2000,
        }

    def _readiness_lock(self) -> dict:
        phase_lock_raw = (self.pre_dir / "phase_lock.jsonl").read_bytes()
        return {
            "artifact_snapshot_sha256": driver.common.sha256_bytes(
                self.artifact_raw
            ),
            "event_ns": self.acquisition_started - 900,
            "phase": "A_ONLY",
            "phase_id": self.phase_id,
            "schema": "s39-cp0-r1-readiness-lock-v2.3",
            "v2_2_phase_lock_sha256": driver.common.sha256_bytes(
                phase_lock_raw
            ),
        }

    @staticmethod
    def _phone_fresh(serial: str, model: str, device: str, boot: str, ip: str):
        return {
            "available_bytes": 2_000_000_000,
            "boot_id": boot,
            "device": device,
            "interfaces": {
                "wlan0": {
                    "ipv4": ip,
                    "rx_bytes": 1000,
                    "tx_bytes": 2000,
                },
            },
            "model": model,
            "product": model,
            "serial": serial,
            "swap_total_bytes": 0,
            "swap_used_bytes": 0,
            "thermal_status": 0,
        }

    def _fresh(self) -> dict:
        return {
            "artifact_stats": [
                {
                    "endpoint": endpoint,
                    "path": self.paths[endpoint],
                    "stat": copy.deepcopy(self.stats[endpoint]),
                }
                for endpoint in (
                    "cuda",
                    "op15",
                    "op12",
                    "op15_worker",
                    "op12_worker",
                )
            ],
            "completed_ns": self.acquisition_started - 10,
            "cuda": {
                "host": "zhihao-Z690-C-ac",
                "host_boot_id": "11111111-1111-4111-8111-111111111111",
                "memory_total_bytes": 17175674880,
                "name": "NVIDIA GeForce RTX 4060 Ti",
                "pci_bus_id": "00000000:01:00.0",
                "uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            },
            "phase": "A_ONLY",
            "phase_id": self.phase_id,
            "phones": {
                "op15": self._phone_fresh(
                    "3C15AU002CL00000",
                    "CPH2749",
                    "OP611FL1",
                    "21111111-1111-4111-8111-111111111111",
                    "192.0.2.15",
                ),
                "op12": self._phone_fresh(
                    "5ae7a43d",
                    "CPH2583",
                    "OP595DL1",
                    "31111111-1111-4111-8111-111111111111",
                    "192.0.2.12",
                ),
            },
            "readiness_lock_sha256": driver.common.sha256_bytes(
                self.readiness_raw
            ),
            "schema": "s39-cp0-r1-fresh-identity-v2.3",
            "started_ns": self.acquisition_started - 20,
        }

    def _plan(self) -> dict:
        producer = self.root / "producer.py"
        producer.write_text(
            "#!/usr/bin/env python3\n"
            "import argparse\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--output')\n"
            "    parser.add_argument('--phase-id')\n"
            "    parser.add_argument('--pre-dir')\n"
            "    parser.add_argument('--acquisition-started-ns')\n"
            "    parser.add_argument('--command-plan-sha256')\n"
            "    parser.parse_args()\n"
            "    return 0\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n",
            encoding="ascii",
        )
        producer.chmod(0o755)
        binding = {
            "argv_index": 0,
            "bytes": producer.stat().st_size,
            "path": str(producer),
            "sha256": driver.common.sha256_bytes(producer.read_bytes()),
        }
        template = [
            str(producer),
            "--output",
            "{output_path}",
            "--phase-id",
            "{phase_id}",
            "--pre-dir",
            "{pre_dir}",
            "--acquisition-started-ns",
            "{acquisition_started_ns}",
            "--command-plan-sha256",
            "{command_plan_sha256}",
        ]
        mechanism = {
            "desktop": [["cuda-route"], ["cuda-monolithic"], ["nvidia-smi"]],
            "op15": [["llama-layersplit"], ["llama-stage-direct-relay"]],
            "op12": [["llama-layersplit"]],
        }
        return {
            "candidate_sha256": driver.common.sha256_bytes(
                driver.v23.DEFAULT_CANDIDATE.read_bytes()
            ),
            "contract_sha256": driver.common.sha256_bytes(
                driver.v23.DEFAULT_CONTRACT.read_bytes()
            ),
            "mechanism_commands": mechanism,
            "model_id": driver.MODEL_ID,
            "model_sha256": self.model["artifact"]["sha256"],
            "outputs": {
                **driver.OUTPUT_FILES,
                "runtime_identity": driver.RUNTIME_FILE,
            },
            "phase": "A_ONLY",
            "producers": {
                "joint_phone_cuda": {
                    "argv_template": copy.deepcopy(template),
                    "executed_files": [copy.deepcopy(binding)],
                    "result_filename": "joint.json",
                    "timeout_seconds": 60,
                },
                "cuda_monolithic": {
                    "argv_template": copy.deepcopy(template),
                    "executed_files": [copy.deepcopy(binding)],
                    "result_filename": "monolithic.json",
                    "timeout_seconds": 60,
                },
            },
            "schema": driver.PLAN_SCHEMA,
        }

    @staticmethod
    def _body(rows: list[dict]) -> list[dict]:
        return [
            {
                key: copy.deepcopy(value)
                for key, value in row.items()
                if key not in driver.COMMON_ROW_KEYS
            }
            for row in rows
        ]

    def _phone_runtime(
        self,
        phone: str,
        local: str,
        peer: str,
        pid: int,
        direct_bytes: int,
    ) -> dict:
        fresh = self.fresh["phones"][phone]
        return {
            "active_sequences_after_cleanup": 0,
            "available_bytes": 1_500_000_000,
            "boot_id": fresh["boot_id"],
            "direct_peer": {
                "interface": "wlan0",
                "local_ipv4": local,
                "peer_ipv4": peer,
                "socket_peer_observed": True,
            },
            "gpu_max_millic": 65000,
            "interface_after": {
                "interface": "wlan0",
                "rx_bytes": 1000 + direct_bytes,
                "tx_bytes": 2000 + direct_bytes,
            },
            "interface_before": {
                "interface": "wlan0",
                "rx_bytes": 1000,
                "tx_bytes": 2000,
            },
            "loaded_shard_path": self.paths[phone],
            "model_id": driver.MODEL_ID,
            "process_swap_bytes": 0,
            "route_epoch": 7,
            "serial": fresh["serial"],
            "session_protocol_version": 2,
            "worker_boot_nonce": "0123456789abcdef",
            "worker_executable_path": self.paths[f"{phone}_worker"],
            "worker_model_sha256": self.model["artifact"]["sha256"],
            "worker_pid": pid,
            "worker_start_ticks": 123456 + pid,
        }

    def _fragments(self) -> list[dict]:
        prefix = f"model.{driver.MODEL_ID}"
        rows = copy.deepcopy(self.phase.rows)
        monolithic = rows[f"{prefix}.oracle.cuda_monolithic"]
        for index, row in enumerate(monolithic):
            row["event_ns"] = self.monolithic_start + 1 + index
        direct_bytes = sum(
            row["payload_bytes"]
            for row in rows[f"{prefix}.route_transfer"]
            if row["kind"] == "transfer"
        )
        mechanism_sha = driver.v2.digest_json(self.plan["mechanism_commands"])
        joint = {
            "bridge_rows": self._body(rows[f"{prefix}.bridge"]),
            "completed_ns": self.joint_complete,
            "cuda_memory_rows": self._body(rows[f"{prefix}.cuda_memory"]),
            "cuda_route_rows": self._body(rows[f"{prefix}.oracle.cuda_route"]),
            "gpu_runtime": {
                "artifact_path": self.paths["cuda"],
                "gpu_uuid": self.fresh["cuda"]["uuid"],
                "host_boot_id": self.fresh["cuda"]["host_boot_id"],
                "model_id": driver.MODEL_ID,
                "route_epoch": 7,
            },
            "mechanism_commands_sha256": mechanism_sha,
            "model_id": driver.MODEL_ID,
            "model_sha256": self.model["artifact"]["sha256"],
            "op12_runtime": self._phone_runtime(
                "op12", "192.0.2.12", "192.0.2.15", 102, direct_bytes
            ),
            "op15_runtime": self._phone_runtime(
                "op15", "192.0.2.15", "192.0.2.12", 101, direct_bytes
            ),
            "phase_id": self.phase_id,
            "mechanics_rows": self._body(
                rows[f"{prefix}.mechanics.phone"]
            ),
            "placement_op12_rows": self._body(
                rows[f"{prefix}.placement.op12"]
            ),
            "placement_op15_rows": self._body(
                rows[f"{prefix}.placement.op15"]
            ),
            "quality_cuda_rows": self._body(rows[f"{prefix}.quality.cuda"]),
            "quality_phone_rows": self._body(rows[f"{prefix}.quality.phone"]),
            "route_transfer_rows": self._body(
                rows[f"{prefix}.route_transfer"]
            ),
            "route_epoch": 7,
            "schema": driver.JOINT_SCHEMA,
            "started_ns": self.start,
        }
        monolithic_fragment = {
            "completed_ns": self.monolithic_complete,
            "mechanism_commands_sha256": mechanism_sha,
            "model_id": driver.MODEL_ID,
            "model_sha256": self.model["artifact"]["sha256"],
            "oracle_cuda_monolithic_rows": self._body(monolithic),
            "phase_id": self.phase_id,
            "schema": driver.MONOLITHIC_SCHEMA,
            "started_ns": self.monolithic_start,
        }
        return [joint, monolithic_fragment]

    def run(self, fragments: list[dict] | None = None):
        runner = FakeRunner(copy.deepcopy(fragments or self.fragments))
        result = driver.acquire(
            self.plan_path.resolve(),
            self.output_dir.resolve(),
            self.phase_id,
            self.pre_dir.resolve(),
            self.acquisition_started,
            runner=runner,
            now_ns=ScriptedClock([
                self.start,
                self.joint_complete + 1,
                self.monolithic_start,
                self.monolithic_complete + 1,
            ]),
        )
        return result, runner


class AOnlyAcquisitionDriverTests(unittest.TestCase):
    def fixture(self):
        return AcquisitionFixture(self)

    def test_valid_fragments_emit_all_roles_and_runtime(self):
        fixture = self.fixture()
        result, runner = fixture.run()
        self.assertEqual(
            result["status"],
            "RAW_ROLES_EMITTED_PENDING_OUTER_V2_3_VALIDATION",
        )
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(runner.calls[0][1:3], ["-I", "-B"])
        for filename in (*driver.OUTPUT_FILES.values(), driver.RUNTIME_FILE):
            self.assertTrue((fixture.output_dir / filename).is_file(), filename)

    def test_missing_raw_role_is_rejected_before_publication(self):
        fixture = self.fixture()
        fragments = copy.deepcopy(fixture.fragments)
        fragments[0]["placement_op12_rows"] = []
        with self.assertRaisesRegex(driver.common.ReadinessError, "E_ROWS"):
            fixture.run(fragments)
        self.assertFalse((fixture.output_dir / "mechanics-phone.jsonl").exists())

    def test_wrong_cross_link_is_rejected(self):
        fixture = self.fixture()
        fragments = copy.deepcopy(fixture.fragments)
        fragments[0]["bridge_rows"][1]["phone_request_sha256"] = "0" * 64
        with self.assertRaisesRegex(driver.v2.EvidenceError, "E_BRIDGE_LINK"):
            fixture.run(fragments)

    def test_swap_in_fresh_snapshot_is_not_weakened(self):
        fixture = self.fixture()
        fixture.fresh["phones"]["op15"]["swap_used_bytes"] = 4096
        (fixture.fresh_dir / "fresh_snapshot.json").write_bytes(
            driver.common.canonical_bytes(fixture.fresh)
        )
        with self.assertRaisesRegex(driver.common.ReadinessError, "PHONE_SWAP"):
            fixture.run()

    def test_uncaptured_repo_import_is_rejected(self):
        fixture = self.fixture()
        producer = Path(
            fixture.plan["producers"]["joint_phone_cuda"]["argv_template"][0]
        )
        producer.write_text(
            "#!/usr/bin/env python3\nimport stage_v3_client\n",
            encoding="ascii",
        )
        binding = fixture.plan["producers"]["joint_phone_cuda"][
            "executed_files"
        ][0]
        binding["bytes"] = producer.stat().st_size
        binding["sha256"] = driver.common.sha256_bytes(producer.read_bytes())
        fixture.plan_path.write_bytes(driver.common.canonical_bytes(fixture.plan))
        with self.assertRaisesRegex(driver.common.ReadinessError, "PRODUCER_IMPORT"):
            fixture.run()

    def test_stale_pyc_cannot_replace_captured_source(self):
        fixture = self.fixture()
        producer = Path(
            fixture.plan["producers"]["joint_phone_cuda"]["argv_template"][0]
        )
        cache = producer.parent / "__pycache__"
        cache.mkdir()
        (cache / "producer.cpython-313.pyc").write_bytes(b"malicious")
        _, runner = fixture.run()
        self.assertEqual(runner.calls[0][1:3], ["-I", "-B"])
        self.assertIn(str(producer.name), runner.calls[0][3])

    def test_changed_producer_source_is_rejected_before_execution(self):
        fixture = self.fixture()
        producer = Path(
            fixture.plan["producers"]["joint_phone_cuda"]["argv_template"][0]
        )
        producer.write_bytes(producer.read_bytes() + b"# changed\n")
        with self.assertRaisesRegex(driver.common.ReadinessError, "EXECUTED_BYTES"):
            fixture.run()


if __name__ == "__main__":
    unittest.main()
