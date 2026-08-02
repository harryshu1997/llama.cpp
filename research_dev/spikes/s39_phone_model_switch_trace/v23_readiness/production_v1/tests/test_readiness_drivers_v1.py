#!/usr/bin/env python3

import copy
import datetime
import hashlib
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
PRODUCTION = HERE.parent
V23 = PRODUCTION.parent
S39 = V23.parent
sys.path.insert(0, str(PRODUCTION))
sys.path.insert(0, str(V23))
sys.path.insert(0, str(S39))

import driver_common_v1 as driver
import cp0_r1_evidence_v23 as v23_validator


class TickingClock:
    def __init__(self, *values):
        self.values = list(values)

    def __call__(self):
        if not self.values:
            raise AssertionError("clock exhausted")
        return self.values.pop(0)


class QueueRunner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def run(self, argv, *, timeout):
        self.calls.append((list(argv), timeout))
        if not self.outputs:
            raise AssertionError(f"unexpected probe: {argv}")
        value = self.outputs.pop(0)
        if isinstance(value, subprocess.CompletedProcess):
            return value
        return subprocess.CompletedProcess(argv, 0, value, b"")


def canonical(value):
    return driver.canonical_bytes(value)


class ReadinessDriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.pre = self.root / "pre"
        self.artifact_out = self.root / "artifact"
        self.fresh_out = self.root / "fresh"
        for path in (self.pre, self.artifact_out, self.fresh_out):
            path.mkdir()
        self.phase_id = "cp0-r1-v23-a-only-test"
        self.contract_path = S39 / "v23_readiness" / "CP0_R1_EVIDENCE_CONTRACT_V2_3.json"
        self.candidate_path = S39 / "CP0_R1_CANDIDATE.json"
        self.contract, _ = driver.read_canonical(self.contract_path)
        self.candidate, _ = driver.read_canonical(self.candidate_path)
        self.model = next(model for model in self.candidate["models"] if model["slot"] == "A")
        geometry = self.contract["model_geometry"][driver.MODEL_ID]
        incumbent = self.contract["incumbent_route_lock"]
        self.route = {
            "acquisition_id": self.phase_id,
            "activation_dtype": geometry["activation_dtype"],
            "activation_element_bytes": geometry["activation_element_bytes"],
            "backend": incumbent["backend"],
            "batch_config_sha256": "a" * 64,
            "clock_id": "HOST_MONOTONIC_RAW",
            "cuda_model_path": geometry["cuda_model_path"],
            "cut_layer": incumbent["cut_layer"],
            "event_ns": 10,
            "frozen_ns": 10,
            "hidden_size": geometry["hidden_size"],
            "kind": "route_lock",
            "model_id": driver.MODEL_ID,
            "model_sha256": self.model["artifact"]["sha256"],
            "n_layer": self.model["n_layer"],
            "op12_shard_bytes": geometry["known_shards"]["op12"]["bytes"],
            "op12_shard_path": geometry["known_shards"]["op12"]["path"],
            "op12_shard_sha256": incumbent["op12_shard_sha256"],
            "op12_stored_layers": incumbent["op12_stored_layers"],
            "op15_shard_bytes": geometry["known_shards"]["op15"]["bytes"],
            "op15_shard_path": geometry["known_shards"]["op15"]["path"],
            "op15_shard_sha256": incumbent["op15_shard_sha256"],
            "op15_stored_layers": incumbent["op15_stored_layers"],
            "phase": "A_ONLY",
            "phase_id": self.phase_id,
            "role": f"model.{driver.MODEL_ID}.route_lock",
        }
        self.route_raw = canonical(self.route)
        (self.pre / "route_lock.jsonl").write_bytes(self.route_raw)
        self.phase_lock = {
            "acquisition_id": self.phase_id,
            "candidate_sha256": "b" * 64,
            "clock_id": "HOST_MONOTONIC_RAW",
            "contract_sha256": "c" * 64,
            "event_ns": 20,
            "kind": "phase_lock",
            "model_slot": "A",
            "phase": "A_ONLY",
            "phase_id": self.phase_id,
            "prior_phase_result_sha256s": [],
            "quality_corpus_sha256": "d" * 64,
            "role": "phase.lock",
            "route_lock_sha256": hashlib.sha256(self.route_raw).hexdigest(),
        }
        (self.pre / "phase_lock.jsonl").write_bytes(canonical(self.phase_lock))
        self.op15_worker = "/data/local/tmp/s39-v23/llama-layersplit"
        self.op12_worker = "/data/local/tmp/s39-v23/llama-layersplit"
        self.paths = {
            "cuda": self.route["cuda_model_path"],
            "op15": self.route["op15_shard_path"],
            "op12": self.route["op12_shard_path"],
            "op15_worker": self.op15_worker,
            "op12_worker": self.op12_worker,
        }
        self.digests = {
            "cuda": self.model["artifact"]["sha256"],
            "op15": self.route["op15_shard_sha256"],
            "op12": self.route["op12_shard_sha256"],
            "op15_worker": "1" * 64,
            "op12_worker": "2" * 64,
        }
        self.stats = {}
        for index, endpoint in enumerate(driver.ENDPOINTS):
            if endpoint == "cuda":
                size = self.model["artifact"]["bytes"]
            elif endpoint in ("op15", "op12"):
                size = self.route[f"{endpoint}_shard_bytes"]
            else:
                size = 1_000_000 + index
            self.stats[endpoint] = {
                "ctime_ns": 1_700_000_100_123_456_789 + index,
                "device_id": 10 + index,
                "inode": 100 + index,
                "mode": stat.S_IFREG | 0o644,
                "mtime_ns": 1_700_000_000_123_456_789 + index,
                "size": size,
            }

    @staticmethod
    def remote_record(record, checksum=None):
        value = {"stat": record}
        if checksum is not None:
            value["sha256"] = checksum
        return canonical(value)

    @staticmethod
    def android_block(record, checksum=None, after=None):
        def time_text(value):
            seconds, nanos = divmod(value, 1_000_000_000)
            parsed = datetime.datetime.fromtimestamp(
                seconds,
                datetime.timezone.utc,
            )
            return parsed.strftime("%Y-%m-%d %H:%M:%S.") + f"{nanos:09d} +0000"

        def block(value):
            return "|".join([
                f"DEV={value['device_id']}",
                f"INO={value['inode']}",
                f"SIZE={value['size']}",
                f"MODE={value['mode']:x}",
                f"MTIME_S={value['mtime_ns'] // 1_000_000_000}",
                f"MTIME={time_text(value['mtime_ns'])}",
                f"CTIME_S={value['ctime_ns'] // 1_000_000_000}",
                f"CTIME={time_text(value['ctime_ns'])}",
            ])

        lines = ["BEFORE", block(record)]
        if checksum is not None:
            lines.extend(["DIGEST=" + checksum, "AFTER", block(after or record)])
        return ("\n".join(lines) + "\n").encode("ascii")

    def artifact_outputs(self):
        result = []
        for endpoint in driver.ENDPOINTS:
            if endpoint == "cuda":
                result.append(
                    self.remote_record(self.stats[endpoint], self.digests[endpoint])
                )
            else:
                result.append(
                    self.android_block(
                        self.stats[endpoint],
                        self.digests[endpoint],
                    )
                )
        return result

    def fresh_outputs(self):
        result = []
        for endpoint in driver.ENDPOINTS:
            if endpoint == "cuda":
                result.append(self.remote_record(self.stats[endpoint]))
            else:
                result.append(self.android_block(self.stats[endpoint]))
        result.append(
            (
                "zhihao-Z690-C-ac\n"
                "11111111-1111-4111-8111-111111111111\n"
                "NVIDIA GeForce RTX 4060 Ti, "
                "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08, "
                "16380, 00000000:01:00.0\n"
            ).encode("ascii")
        )
        result.extend(
            [
                self.phone_output(
                    "3C15AU002CL00000",
                    "CPH2749",
                    "OP611FL1",
                    "21111111-1111-4111-8111-111111111111",
                    "192.0.2.15",
                ),
                self.phone_output(
                    "5ae7a43d",
                    "CPH2583",
                    "OP595DL1",
                    "31111111-1111-4111-8111-111111111111",
                    "192.0.2.12",
                ),
            ]
        )
        return result

    @staticmethod
    def phone_output(
        serial,
        model,
        device,
        boot_id,
        ipv4,
        *,
        swap_free=0,
        swap_total=0,
        thermal=0,
    ):
        return (
            f"SERIAL={serial}\n"
            f"MODEL={model}\n"
            f"PRODUCT={model}\n"
            f"DEVICE={device}\n"
            f"BOOT_ID={boot_id}\n"
            "MEM_AVAILABLE_KB=2000000\n"
            f"SWAP_TOTAL_KB={swap_total}\n"
            f"SWAP_FREE_KB={swap_free}\n"
            f"THERMAL_STATUS={thermal}\n"
            f"IF=wlan0|{ipv4}|1000|2000\n"
        ).encode("ascii")

    def run_artifact(self, runner=None):
        runner = runner or QueueRunner(self.artifact_outputs())
        result = driver.artifact_driver(
            contract_path=self.contract_path,
            candidate_path=self.candidate_path,
            pre_dir=self.pre,
            output_dir=self.artifact_out,
            phase_id=self.phase_id,
            op15_worker=self.op15_worker,
            op12_worker=self.op12_worker,
            timeout=30,
            runner=runner,
            now_ns=TickingClock(100, 200),
        )
        return result, runner

    def run_fresh(self, outputs=None):
        runner = QueueRunner(outputs or self.fresh_outputs())
        result = driver.fresh_driver(
            contract_path=self.contract_path,
            candidate_path=self.candidate_path,
            pre_dir=self.pre,
            output_dir=self.fresh_out,
            phase_id=self.phase_id,
            op15_worker=self.op15_worker,
            op12_worker=self.op12_worker,
            timeout=30,
            runner=runner,
            now_ns=TickingClock(300, 310, 400),
        )
        return result, runner

    def test_artifact_snapshot_is_exact_and_digest_bound(self):
        result, runner = self.run_artifact()
        self.assertEqual(result["schema"], "s39-cp0-r1-artifact-snapshot-v2.3")
        self.assertEqual(
            [value["endpoint"] for value in result["artifacts"]],
            list(driver.ENDPOINTS),
        )
        self.assertEqual(len(runner.calls), 5)
        self.assertEqual(runner.calls[0][0][0], "ssh")
        self.assertEqual(
            runner.calls[1][0][:5],
            ["adb", "-P", "5038", "-s", "172.20.173.218:5555"],
        )
        self.assertEqual(
            runner.calls[2][0][:5],
            ["adb", "-P", "5038", "-s", "172.20.59.72:5555"],
        )
        self.assertEqual(runner.calls[3][0][:5], runner.calls[1][0][:5])
        self.assertEqual(runner.calls[4][0][:5], runner.calls[2][0][:5])
        raw = (self.artifact_out / "artifact_snapshot.json").read_bytes()
        self.assertEqual(raw, canonical(result))

    def test_artifact_digest_mismatch_refuses_without_output(self):
        outputs = self.artifact_outputs()
        value = driver.parse_json(outputs[0], "cuda")
        value["sha256"] = "0" * 64
        outputs[0] = canonical(value)
        with self.assertRaisesRegex(driver.DriverError, "E_SHA256"):
            self.run_artifact(QueueRunner(outputs))
        self.assertFalse((self.artifact_out / "artifact_snapshot.json").exists())

    def test_artifact_changed_during_hash_is_rejected(self):
        outputs = self.artifact_outputs()
        changed = copy.deepcopy(self.stats["op15"])
        changed["ctime_ns"] += 1
        outputs[1] = self.android_block(
            self.stats["op15"],
            self.digests["op15"],
            changed,
        )
        with self.assertRaisesRegex(driver.DriverError, "CHANGED_DURING_HASH"):
            self.run_artifact(QueueRunner(outputs))

    def test_android_stat_command_matches_real_stat(self):
        path = self.root / "stat-input.bin"
        path.write_bytes(b"phone-artifact")
        completed = subprocess.run(
            ["/bin/sh", "-c", driver.android_stat_command(str(path), True)],
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, b"")
        parsed = driver.parse_android_stat(
            completed.stdout,
            "local-stat",
            True,
        )
        self.assertEqual(
            parsed["sha256"],
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        self.assertEqual(parsed["stat"], driver.stat_record(path.stat()))

    def test_probe_stderr_and_timeout_fail_closed(self):
        for result, message in (
            (subprocess.CompletedProcess([], 0, b"ok\n", b"warning\n"), "STDERR"),
            (subprocess.CompletedProcess([], 9, b"ok\n", b""), "EXIT"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(driver.DriverError, message):
                    self.run_artifact(QueueRunner([result]))

    def test_fresh_snapshot_rechecks_artifacts_and_identity(self):
        artifact, _ = self.run_artifact()
        (lock, fresh), runner = self.run_fresh()
        self.assertEqual(lock["event_ns"], 300)
        self.assertEqual(fresh["started_ns"], 310)
        self.assertEqual(fresh["completed_ns"], 400)
        self.assertEqual(
            [row["stat"] for row in fresh["artifact_stats"]],
            [row["stat"] for row in artifact["artifacts"]],
        )
        self.assertEqual(fresh["phones"]["op15"]["serial"], "3C15AU002CL00000")
        self.assertEqual(fresh["phones"]["op12"]["serial"], "5ae7a43d")
        self.assertEqual(fresh["phones"]["op15"]["interfaces"]["wlan0"]["ipv4"], "192.0.2.15")
        self.assertEqual(len(runner.calls), 8)
        self.assertEqual(
            runner.calls[6][0][:5],
            ["adb", "-P", "5038", "-s", "172.20.173.218:5555"],
        )
        self.assertEqual(
            runner.calls[7][0][:5],
            ["adb", "-P", "5038", "-s", "172.20.59.72:5555"],
        )
        self.assertTrue((self.fresh_out / "readiness_lock.json").is_file())
        self.assertTrue((self.fresh_out / "fresh_snapshot.json").is_file())
        artifacts = v23_validator.validate_artifact_snapshot(
            artifact,
            "A_ONLY",
            self.model,
            self.route,
        )
        manifest = {
            "acquisition_started_ns": 500,
            "phase": "A_ONLY",
            "phase_id": self.phase_id,
        }
        v23_validator.validate_readiness_lock(
            lock,
            manifest,
            hashlib.sha256(canonical(self.phase_lock)).hexdigest(),
            canonical(artifact),
            artifact["completed_ns"],
        )
        derived = v23_validator.validate_fresh_snapshot(
            fresh,
            canonical(fresh),
            canonical(lock),
            lock,
            manifest,
            self.contract,
            artifacts,
        )
        self.assertEqual(derived["cuda"]["uuid"], fresh["cuda"]["uuid"])

    def test_fresh_rejects_changed_artifact(self):
        self.run_artifact()
        outputs = self.fresh_outputs()
        changed = copy.deepcopy(self.stats["op12"])
        changed["mtime_ns"] += 1
        outputs[2] = self.android_block(changed)
        with self.assertRaisesRegex(driver.DriverError, "ARTIFACT_CHANGED"):
            self.run_fresh(outputs)
        self.assertTrue((self.fresh_out / "readiness_lock.json").is_file())
        self.assertFalse((self.fresh_out / "fresh_snapshot.json").exists())

    def test_fresh_rejects_wrong_gpu(self):
        self.run_artifact()
        outputs = self.fresh_outputs()
        outputs[5] = outputs[5].replace(
            b"GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            b"GPU-00000000-0000-0000-0000-000000000000",
        )
        with self.assertRaisesRegex(driver.DriverError, "CUDA_UUID"):
            self.run_fresh(outputs)

    def test_fresh_rejects_wifi_selector_to_physical_serial_swap(self):
        self.run_artifact()
        outputs = self.fresh_outputs()
        outputs[6], outputs[7] = outputs[7], outputs[6]
        with self.assertRaisesRegex(driver.DriverError, "PHONE_IDENTITY"):
            self.run_fresh(outputs)

    def test_phone_identity_requires_selector_and_physical_serial(self):
        self.assertIn("getprop ro.serialno", driver.phone_identity_command())
        expected = copy.deepcopy(
            self.contract["readiness_v2_3"]["phone_identity"]["op15"]
        )
        raw = self.phone_output(
            "3C15AU002CL00000",
            "CPH2749",
            "OP611FL1",
            "21111111-1111-4111-8111-111111111111",
            "192.0.2.15",
        )
        del expected["adb_selector"]
        with self.assertRaisesRegex(driver.DriverError, "E_KEYS"):
            driver.parse_phone_identity(raw, "3C15AU002CL00000", expected)

        expected = copy.deepcopy(
            self.contract["readiness_v2_3"]["phone_identity"]["op15"]
        )
        wrong = raw.replace(b"SERIAL=3C15AU002CL00000", b"SERIAL=5ae7a43d")
        with self.assertRaisesRegex(driver.DriverError, "PHONE_IDENTITY"):
            driver.parse_phone_identity(
                wrong,
                "3C15AU002CL00000",
                expected,
            )

    def test_fresh_rejects_phone_swap_thermal_and_headroom(self):
        mutations = (
            (
                self.phone_output(
                    "3C15AU002CL00000",
                    "CPH2749",
                    "OP611FL1",
                    "21111111-1111-4111-8111-111111111111",
                    "192.0.2.15",
                    swap_total=10,
                ),
                "PHONE_SWAP",
            ),
            (
                self.phone_output(
                    "3C15AU002CL00000",
                    "CPH2749",
                    "OP611FL1",
                    "21111111-1111-4111-8111-111111111111",
                    "192.0.2.15",
                    thermal=2,
                ),
                "PHONE_THERMAL",
            ),
            (
                self.phone_output(
                    "3C15AU002CL00000",
                    "CPH2749",
                    "OP611FL1",
                    "21111111-1111-4111-8111-111111111111",
                    "192.0.2.15",
                ).replace(b"MEM_AVAILABLE_KB=2000000", b"MEM_AVAILABLE_KB=1"),
                "PHONE_HEADROOM",
            ),
        )
        for index, (replacement, message) in enumerate(mutations):
            with self.subTest(index=index):
                if index:
                    shutil.rmtree(self.fresh_out)
                    self.fresh_out.mkdir()
                if not (self.artifact_out / "artifact_snapshot.json").exists():
                    self.run_artifact()
                outputs = self.fresh_outputs()
                outputs[6] = replacement
                with self.assertRaisesRegex(driver.DriverError, message):
                    self.run_fresh(outputs)

    def test_symlinked_contract_is_rejected(self):
        link = self.root / "contract-link.json"
        link.symlink_to(self.contract_path)
        with self.assertRaisesRegex(driver.DriverError, "E_READ"):
            driver.artifact_driver(
                contract_path=link,
                candidate_path=self.candidate_path,
                pre_dir=self.pre,
                output_dir=self.artifact_out,
                phase_id=self.phase_id,
                op15_worker=self.op15_worker,
                op12_worker=self.op12_worker,
                timeout=30,
                runner=QueueRunner([]),
            )

    def test_copied_entrypoint_loads_explicit_support(self):
        entry = self.root / "captured-artifact"
        support = self.root / "captured-support.py"
        shutil.copyfile(PRODUCTION / "artifact_snapshot_driver_v1.py", entry)
        shutil.copyfile(PRODUCTION / "driver_common_v1.py", support)
        entry.chmod(0o755)
        completed = subprocess.run(
            [str(entry), "--support", str(support), "--help"],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("--phase-id", completed.stdout)
        self.assertEqual(completed.stderr, "")

    def test_cli_missing_support_refuses_without_traceback(self):
        entry = PRODUCTION / "fresh_readiness_driver_v1.py"
        completed = subprocess.run(
            [sys.executable, str(entry)],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("missing --support", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
