#!/usr/bin/env python3

import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent
V23 = DRIVER.parent
S39 = V23.parent
sys.path[:0] = [
    str(DRIVER),
    str(V23),
    str(V23 / "tests"),
    str(S39),
    str(S39 / "tests"),
]

import acquisition_support_v1 as acquisition
import cp0_r1_evidence_v2 as v2
import cp0_r1_evidence_v22 as v22
import cp0_r1_evidence_v23 as v23
import v23_common as common
from test_v23_readiness import ReadinessFixture


COMMON = {"acquisition_id", "phase", "phase_id", "role"}


class SequenceClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        if not self.values:
            raise AssertionError("clock exhausted")
        return self.values.pop(0)


class FakeRunner:
    def __init__(self, results, *, fail=None):
        self.results = results
        self.fail = fail
        self.calls = []

    def run(self, argv, *, timeout):
        del timeout
        name = (
            "joint_phone_cuda"
            if any(item.endswith("/joint.json") for item in argv)
            else "cuda_monolithic"
        )
        self.calls.append((name, list(argv)))
        output = Path(argv[argv.index("--output") + 1])
        output.write_bytes(common.canonical_bytes(self.results[name]))
        return subprocess.CompletedProcess(
            argv,
            2 if name == self.fail else 0,
            b"",
            b"",
        )


class AcquisitionSupportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.phase_id = "cp0-r1-v23-a-only-driver-test"
        self.fixture = ReadinessFixture(self)
        phase = self.fixture.phase
        phase.set_phase_id(self.phase_id, write=False)
        phase.repair_model()
        phase.write()
        self.fixture.manifest, _ = v23.load_v22_manifest(phase.root)
        self.fixture.route = phase.lock
        self.fixture.artifact = self.fixture.make_artifact()
        phase_lock_sha = next(
            item["sha256"]
            for item in self.fixture.manifest["artifacts"]
            if item["role"] == "phase.lock"
        )
        self.fixture.lock = {
            "artifact_snapshot_sha256": common.sha256_bytes(
                common.canonical_bytes(self.fixture.artifact)
            ),
            "event_ns": phase.started_ns - 400,
            "phase": "A_ONLY",
            "phase_id": self.phase_id,
            "schema": "s39-cp0-r1-readiness-lock-v2.3",
            "v2_2_phase_lock_sha256": phase_lock_sha,
        }
        self.fixture.lock_raw = common.canonical_bytes(self.fixture.lock)
        self.fixture.fresh = self.fixture.make_fresh()
        self.fixture.fresh_raw = common.canonical_bytes(self.fixture.fresh)

        self.outer = self.root / "outer"
        self.pre = self.outer / "pre"
        self.artifact_dir = self.outer / "artifact"
        self.fresh_dir = self.outer / "fresh"
        self.output = self.outer / "acquisition"
        for path in (self.pre, self.artifact_dir, self.fresh_dir, self.output):
            path.mkdir(parents=True)
        prefix = f"model.{acquisition.MODEL_ID}"
        for name, rows in (
            ("route_lock.jsonl", phase.rows[f"{prefix}.route_lock"]),
            ("quality_corpus.jsonl", phase.rows["quality.corpus"]),
            ("phase_lock.jsonl", phase.rows["phase.lock"]),
        ):
            (self.pre / name).write_bytes(
                b"".join(v2.canonical_line(row) for row in rows)
            )
        (self.artifact_dir / "artifact_snapshot.json").write_bytes(
            common.canonical_bytes(self.fixture.artifact)
        )
        (self.fresh_dir / "readiness_lock.json").write_bytes(
            common.canonical_bytes(self.fixture.lock)
        )
        (self.fresh_dir / "fresh_snapshot.json").write_bytes(
            common.canonical_bytes(self.fixture.fresh)
        )

        self.mechanisms = {
            "desktop": [["/bin/true"], ["/bin/true"]],
            "op12": [["/bin/true"]],
            "op15": [["/bin/true"], ["/bin/true"]],
        }
        self.mechanism_sha = v2.digest_json(self.mechanisms)
        self.results = self.make_results()
        self.fake = self.root / "fake_producer.py"
        self.fake.write_text(
            "#!/usr/bin/python3 -I\n"
            "import sys\n"
            "raise SystemExit(90)\n",
            encoding="ascii",
        )
        self.fake.chmod(0o755)
        binding = {
            "argv_index": 0,
            "bytes": self.fake.stat().st_size,
            "path": str(self.fake),
            "sha256": hashlib.sha256(self.fake.read_bytes()).hexdigest(),
        }
        contract_raw = v23.DEFAULT_CONTRACT.read_bytes()
        candidate_raw = v23.DEFAULT_CANDIDATE.read_bytes()
        self.plan = {
            "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
            "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
            "mechanism_commands": self.mechanisms,
            "model_id": acquisition.MODEL_ID,
            "model_sha256": self.fixture.model["artifact"]["sha256"],
            "outputs": {
                **acquisition.OUTPUT_FILES,
                "runtime_identity": acquisition.RUNTIME_FILE,
            },
            "phase": "A_ONLY",
            "producers": {
                "joint_phone_cuda": self.producer(binding, "joint.json"),
                "cuda_monolithic": self.producer(binding, "monolithic.json"),
            },
            "schema": acquisition.PLAN_SCHEMA,
        }
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_bytes(common.canonical_bytes(self.plan))

    def producer(self, binding, filename):
        return {
            "argv_template": [
                str(self.fake),
                "--output",
                "{output_path}",
                "--phase-id",
                "{phase_id}",
                "--pre-dir",
                "{pre_dir}",
                "--started",
                "{acquisition_started_ns}",
                "--plan",
                "{command_plan_sha256}",
            ],
            "executed_files": [copy.deepcopy(binding)],
            "result_filename": filename,
            "timeout_seconds": 60,
        }

    @staticmethod
    def bodies(rows):
        return [
            {key: copy.deepcopy(value) for key, value in row.items() if key not in COMMON}
            for row in rows
        ]

    def runtime_body(self, executor):
        excluded = {
            "executor_id",
            "loaded_shard_sha256",
            "loaded_shard_stat",
            "mechanics_sha256",
            "placement_sha256",
            "route_transfer_sha256",
            "worker_executable_sha256",
            "worker_executable_stat",
        }
        return {
            key: copy.deepcopy(value)
            for key, value in executor.items()
            if key not in excluded
        }

    def make_results(self):
        phase = self.fixture.phase
        prefix = f"model.{acquisition.MODEL_ID}"
        runtime = self.fixture.make_runtime()
        joint_started = phase.started_ns + 1
        joint_completed = phase.started_ns + 1_000_000
        mono_started = phase.started_ns + 2_000_000
        mono_completed = mono_started + 100
        mono_rows = copy.deepcopy(phase.rows[f"{prefix}.oracle.cuda_monolithic"])
        for index, row in enumerate(mono_rows):
            row["event_ns"] = mono_started + index
        joint = {
            "bridge_rows": self.bodies(phase.rows[f"{prefix}.bridge"]),
            "completed_ns": joint_completed,
            "cuda_memory_rows": self.bodies(phase.rows[f"{prefix}.cuda_memory"]),
            "cuda_route_rows": self.bodies(phase.rows[f"{prefix}.oracle.cuda_route"]),
            "gpu_runtime": {
                key: copy.deepcopy(runtime["executors"][0][key])
                for key in acquisition.GPU_RUNTIME_KEYS
            },
            "mechanism_commands_sha256": self.mechanism_sha,
            "mechanics_rows": self.bodies(phase.rows[f"{prefix}.mechanics.phone"]),
            "model_id": acquisition.MODEL_ID,
            "model_sha256": self.fixture.model["artifact"]["sha256"],
            "op12_runtime": self.runtime_body(runtime["executors"][2]),
            "op15_runtime": self.runtime_body(runtime["executors"][1]),
            "phase_id": self.phase_id,
            "placement_op12_rows": self.bodies(
                phase.rows[f"{prefix}.placement.op12"]
            ),
            "placement_op15_rows": self.bodies(
                phase.rows[f"{prefix}.placement.op15"]
            ),
            "quality_cuda_rows": self.bodies(
                phase.rows[f"{prefix}.quality.cuda"]
            ),
            "quality_phone_rows": self.bodies(
                phase.rows[f"{prefix}.quality.phone"]
            ),
            "route_epoch": runtime["route_epoch"],
            "route_transfer_rows": self.bodies(
                phase.rows[f"{prefix}.route_transfer"]
            ),
            "schema": acquisition.JOINT_SCHEMA,
            "started_ns": joint_started,
        }
        monolithic = {
            "completed_ns": mono_completed,
            "mechanism_commands_sha256": self.mechanism_sha,
            "model_id": acquisition.MODEL_ID,
            "model_sha256": self.fixture.model["artifact"]["sha256"],
            "oracle_cuda_monolithic_rows": self.bodies(mono_rows),
            "phase_id": self.phase_id,
            "schema": acquisition.MONOLITHIC_SCHEMA,
            "started_ns": mono_started,
        }
        return {
            "joint_phone_cuda": joint,
            "cuda_monolithic": monolithic,
        }

    def acquire(self, runner=None):
        phase = self.fixture.phase
        clock = SequenceClock([
            phase.started_ns,
            self.results["joint_phone_cuda"]["completed_ns"] + 1,
            self.results["cuda_monolithic"]["started_ns"] - 1,
            self.results["cuda_monolithic"]["completed_ns"] + 1,
        ])
        return acquisition.acquire(
            self.plan_path,
            self.output,
            self.phase_id,
            self.pre,
            phase.started_ns,
            runner=runner or FakeRunner(self.results),
            now_ns=clock,
        )

    def test_fake_commands_emit_all_roles_and_runtime(self):
        result = self.acquire()
        self.assertEqual(
            result["status"],
            "RAW_ROLES_EMITTED_PENDING_OUTER_V2_3_VALIDATION",
        )
        for filename in (*acquisition.OUTPUT_FILES.values(), acquisition.RUNTIME_FILE):
            self.assertTrue((self.output / filename).is_file(), filename)
        self.assertEqual(len(result["output_sha256s"]), 11)

    def test_mechanism_digest_mutation_is_rejected(self):
        self.results["joint_phone_cuda"]["mechanism_commands_sha256"] = "0" * 64
        with self.assertRaisesRegex(common.ReadinessError, "mechanism_commands"):
            self.acquire()

    def test_missing_role_rows_are_rejected_before_publish(self):
        self.results["joint_phone_cuda"]["placement_op12_rows"] = []
        with self.assertRaisesRegex(common.ReadinessError, "ROWS"):
            self.acquire()
        self.assertFalse((self.output / "mechanics-phone.jsonl").exists())

    def test_producer_failure_is_retained_and_refused(self):
        runner = FakeRunner(self.results, fail="joint_phone_cuda")
        with self.assertRaisesRegex(common.ReadinessError, "PRODUCER_EXIT"):
            self.acquire(runner)
        self.assertTrue(
            (self.output / "raw" / "joint_phone_cuda.receipt.json").is_file()
        )

    def test_float_timestamp_is_rejected(self):
        self.results["joint_phone_cuda"]["mechanics_rows"][0]["event_ns"] = 1.0
        with self.assertRaisesRegex(common.ReadinessError, "event_ns"):
            self.acquire()


if __name__ == "__main__":
    unittest.main()
