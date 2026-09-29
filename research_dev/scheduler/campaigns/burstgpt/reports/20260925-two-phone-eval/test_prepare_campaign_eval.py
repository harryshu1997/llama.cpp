"""Unit tests for the fixed-worker receipt checks in prepare_campaign_eval.py.

    python3 -m unittest test_prepare_campaign_eval   (from this directory)

Synthetic trees check that the byte-identity bridge accepts the configured worker only when every link
holds, and rejects each broken link. One test runs the bridge and the calibration on the real evidence
copies in ../20260925-pixel-cpu-gpu/.
"""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import make_gate_config
import prepare_campaign_eval as prep

HERE = Path(__file__).resolve().parent
INT2_GATE = HERE.parent / "20260924-pixel-second-phone/GATE_CONFIG.json"
CPUGPU = HERE.parent / "20260925-pixel-cpu-gpu"
SEGMENTS = {"m1": 1, "m2": 2, "m4": 4}


def config():
    return make_gate_config.derive(json.loads(INT2_GATE.read_text()))


def hexsha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def launch(worker, library, shard, environment):
    assignments = " ".join(f"{key}={value}" for key, value in sorted(environment.items()))
    return (f"su -c 'exec 9>/data/local/tmp/.lock\nflock -n 9 9>&9 || exit 73\nexec env LD_LIBRARY_PATH={library} "
            f"{assignments} {worker} -m {shard} --layers 18-23 --max-tokens 4 --max-requests 270'")


class Fixture:
    """A minimal evidence tree: numerical suite, TCP qualification, phone-local rows-3 and flag suites."""

    def __init__(self, root: Path):
        self.root = root
        gate = config()
        self.configured, self.reference = gate["helper_phone"], gate["helper_phone_qualified_reference"]
        self.staged = self.configured["library_directories"][0]
        self.prod_worker = self.staged + "/prod/llama-ffn-split-worker"
        self.evidence = root / "cpugpu-evidence"
        self.numerical = root / "numerical"
        qualified = {path: value[7:] for path, value in self.reference["expected_sha256_by_path"].items()}
        dump(self.numerical / "EXPECTED_PHONE_HASHES.json", qualified)
        dump(self.numerical / "SUITE_RESULT.json", {"status": "PASS", "arms": {"04-sdot-pair-dynamic64": {
            "numerical_status": "PASS", "max_relative_l2": 5e-4, "arm": {"cpu_pair_dot": 1, "cpu_row_chunk": 64}}}})
        dump(self.numerical / "SUITE_CALLS.json", {"04-sdot-pair-dynamic64": [{"tokens": t} for t in (1, 2, 4, 8)]})
        self.outputs = {segment: bytes([rows]) * 64 for segment, rows in SEGMENTS.items()}
        self.tcp_arms = [self.tcp_arm("a-prod", reference=True), self.tcp_arm("c-boost-batch", reference=False)]
        self.write_tcp_suite()
        self.local_suite("d3", "d3-m3", {"burst-m3": 3, "prod-m3": 3},
                         [("a-prod", True, self.reference["worker_environment"]),
                          ("b-boost-batch", False, self.configured["worker_environment"])])
        flags = dict(self.reference["worker_environment"], S43_PIXEL_UCLAMP_MIN="1024")
        self.local_suite("d1", "d1-cadence", {"burst-m1": 1, "prod-m4": 4},
                         [("a-prod", True, self.reference["worker_environment"]), ("c-uclamp", False, flags),
                          ("g-gpu", False, {"S42_PIXEL_GPU_EXPAND_F16": "1"})])

    def tcp_arm(self, name, *, reference):
        worker = self.prod_worker if reference else self.configured["worker_path"]
        environment = self.reference["worker_environment"] if reference else self.configured["worker_environment"]
        pins = dict(self.configured["expected_sha256_by_path"])
        worker_hash = self.reference["expected_sha256_by_path"][self.reference["worker_path"]] if reference \
            else pins[self.configured["worker_path"]]
        if reference:
            del pins[self.configured["worker_path"]]
            pins[worker] = worker_hash
        directory = self.evidence / "tcp1" / name
        for segment, data in self.outputs.items():
            directory.mkdir(parents=True, exist_ok=True)
            (directory / (segment + ".f16")).write_bytes(data)
        calls = [{"segment": segment, "rows": rows, "step": step, "layer": 18 + layer,
                  "compute_us": 6000 * rows + 100 * layer, "overhead_us": 7000 + 1000 * rows, "rpc_us": 13000 * rows}
                 for segment, rows in SEGMENTS.items() for step in range(-1, 5) for layer in range(6)]
        dump(directory / "CALLS.json", calls)
        dump(directory / "PREFLIGHT.json", {"observed_sha256_by_path": pins, "port_free": True, "worker_absent": True})
        dump(directory / "LAUNCH.json", {"command": ["adb", "shell", "-T", launch(
            worker, self.staged, self.configured["shard_path"], environment)]})
        return {"status": "PASS", "arm": name, "calls": len(calls), "worker_sha256": worker_hash,
                "environment": dict(environment), "dump_sha256": {s: hexsha(d) for s, d in self.outputs.items()},
                "stop": {"exit_code": 0, "signalled": False, "boot_unchanged": True, "forward_removed": True,
                         "worker_pids_after": [], "drained_calls": 0}}

    def write_tcp_suite(self):
        dump(self.evidence / "tcp1/SUITE_RESULT.json", self.tcp_arms)

    def local_suite(self, suite, run, segments, arms):
        rows = []
        for name, prod, environment in arms:
            worker = self.prod_worker if prod else self.configured["worker_path"]
            assignments = " ".join(f"{k}={v}" for k, v in sorted(environment.items()))
            row = {"name": name, "env": environment, "worker_command":
                   f"env LD_LIBRARY_PATH={self.staged} {assignments} {worker} -m {self.configured['shard_path']} --port 1"}
            if prod:
                row["prod"] = True
            if name.startswith("g-"):
                row["backend"] = "Vulkan0"
            rows.append(row)
            directory = self.evidence / suite / run / name
            directory.mkdir(parents=True, exist_ok=True)
            for filename in ("WORKER_EXIT.txt", "CLIENT_EXIT.txt"):
                (directory / filename).write_text("0\n")
            for segment, count in segments.items():
                data = b"gpu" if name.startswith("g-") else bytes([count]) * 32
                (directory / ("replay." + segment + ".f16")).write_bytes(data)
        dump(self.evidence / "suites" / suite / "SUITE.json",
             {"segments": [{"name": s, "rows": r} for s, r in segments.items()], "arms": rows})

    def receipt(self, configured=None):
        return prep.numerical_receipt(self.numerical, configured or self.configured, self.reference, self.evidence)


class ByteIdentityBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def assertRejected(self, message):
        with self.assertRaises(AssertionError) as caught:
            self.fixture.receipt()
        self.assertIn(message, str(caught.exception))

    def test_accepts_byte_identical_fixed_worker(self):
        receipt = self.fixture.receipt()
        bridge = receipt["byte_identity_bridge"]
        self.assertEqual(bridge["configured_worker_sha256"][:15], "sha256:5d824455")
        self.assertEqual(bridge["qualified_worker_sha256"][:15], "sha256:64133753")
        self.assertEqual(bridge["tcp_rows_1_2_4"]["rows"], [1, 2, 4])
        self.assertEqual(bridge["tcp_rows_1_2_4"]["candidate_arms"], ["c-boost-batch"])
        self.assertEqual(bridge["phone_local_rows_3"]["candidate_arms"], ["b-boost-batch"])
        self.assertEqual(bridge["phone_local_flag_matrix"]["identical_cpu_arms"], ["a-prod", "c-uclamp"])

    def test_qualified_worker_needs_no_bridge(self):
        receipt = self.fixture.receipt(self.fixture.reference)
        self.assertNotIn("byte_identity_bridge", receipt)

    def test_rejects_libraries_or_shard_outside_the_suite(self):
        configured = copy.deepcopy(self.fixture.configured)
        configured["expected_sha256_by_path"][configured["shard_path"]] = "sha256:" + "0" * 64
        with self.assertRaises(AssertionError) as caught:
            self.fixture.receipt(configured)
        self.assertIn("libraries or shard", str(caught.exception))

    def test_rejects_reference_worker_outside_the_suite(self):
        path = self.fixture.numerical / "EXPECTED_PHONE_HASHES.json"
        hashes = json.loads(path.read_text())
        hashes.pop(self.fixture.reference["worker_path"])
        dump(path, hashes)
        self.assertRejected("not the numerically qualified one")

    def test_rejects_different_outputs_even_when_recorded_consistently(self):
        arm = self.fixture.tcp_arms[1]
        (self.fixture.evidence / "tcp1/c-boost-batch/m4.f16").write_bytes(b"different")
        self.assertRejected("dump files differ from the recorded hashes")
        arm["dump_sha256"]["m4"] = hexsha(b"different")
        self.fixture.write_tcp_suite()
        self.assertRejected("not byte-identical to the qualified worker")

    def test_rejects_another_environment(self):
        configured = copy.deepcopy(self.fixture.configured)
        configured["worker_environment"]["S43_PIXEL_CPU_POLL"] = "200"
        with self.assertRaises(AssertionError) as caught:
            self.fixture.receipt(configured)
        self.assertIn("no configured or no reference arm", str(caught.exception))

    def test_rejects_launch_that_differs_from_the_recorded_environment(self):
        path = self.fixture.evidence / "tcp1/c-boost-batch/LAUNCH.json"
        command = json.loads(path.read_text())
        command["command"][-1] = command["command"][-1].replace("S43_PIXEL_CPU_BATCH_PAIR=1 ", "")
        dump(path, command)
        self.assertRejected("launch environment differs")

    def test_rejects_unclean_stop(self):
        self.fixture.tcp_arms[1]["stop"]["signalled"] = True
        self.fixture.write_tcp_suite()
        self.assertRejected("stopped uncleanly")

    def test_rejects_preflight_that_differs_from_the_pins(self):
        path = self.fixture.evidence / "tcp1/c-boost-batch/PREFLIGHT.json"
        preflight = json.loads(path.read_text())
        preflight["observed_sha256_by_path"][self.fixture.configured["worker_path"]] = "sha256:" + "1" * 64
        dump(path, preflight)
        self.assertRejected("preflight differs from the configured pins")

    def test_rejects_missing_rows(self):
        arm = self.fixture.tcp_arms[1]
        del arm["dump_sha256"]["m4"]
        self.fixture.tcp_arms[0]["dump_sha256"].pop("m4")
        path = self.fixture.evidence / "tcp1/c-boost-batch/CALLS.json"
        dump(path, [row for row in json.loads(path.read_text()) if row["rows"] != 4])
        self.fixture.write_tcp_suite()
        self.assertRejected("lacks rows 1, 2 or 4")

    def test_rejects_phone_local_row_three_mismatch(self):
        (self.fixture.evidence / "d3/d3-m3/b-boost-batch/replay.prod-m3.f16").write_bytes(b"x")
        self.assertRejected("phone-local outputs are not byte-identical")

    def test_rejects_flag_matrix_mismatch(self):
        (self.fixture.evidence / "d1/d1-cadence/c-uclamp/replay.burst-m1.f16").write_bytes(b"x")
        self.assertRejected("flag-matrix outputs differ")

    def test_rejects_nonzero_phone_local_exit(self):
        (self.fixture.evidence / "d3/d3-m3/b-boost-batch/WORKER_EXIT.txt").write_text("1\n")
        self.assertRejected("exited nonzero")


class ServerIdentityAndIdleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.configured = config()["helper_phone"]
        self.server = self.root / "bin/llama-server"
        self.server.parent.mkdir()
        self.server.write_bytes(b"server")
        (self.server.parent / "libllama-server-impl.so").write_bytes(b"impl")
        run = self.root / "server-identity-r1"
        self.run = run
        c = self.configured
        dump(run / "RESULT.json", {"status": "PASS", "token_identity": [True] * 4, "output_tokens_each": 64,
                                   "server_exit": 0, "worker_exit": 0, "phone_calls": 744,
                                   "call_columns": {"8704": 372, "17408": 372}})
        dump(run / "CONFIG.json", {"phone_worker": c["worker_path"], "phone_model": c["shard_path"],
                                   "phone_library_dir": c["library_directories"][0],
                                   "phone_environment": c["worker_environment"], "phone_root": True})
        lines = "\n".join(value[7:] + "  " + path for path, value in c["expected_sha256_by_path"].items())
        dump(run / "IDENTITY.json", {"phone_hashes": lines, "server_sha256": hexsha(b"server"),
                                     "server_libraries": {str(self.server.parent / "libllama-server-impl.so"): hexsha(b"impl")}})
        dump(run / "WORKER_COMMAND.json", ["adb", "shell", "-T", launch(
            c["worker_path"], c["library_directories"][0], c["shard_path"], c["worker_environment"])])

    def tearDown(self):
        self.temporary.cleanup()

    def test_accepts_fresh_pixel_only_run(self):
        check = prep.server_identity(self.run, self.configured, self.server)
        self.assertEqual(check["outputs"], 4)
        self.assertEqual(check["call_columns"], {"8704": 372, "17408": 372})

    def test_rejects_token_difference(self):
        result = json.loads((self.run / "RESULT.json").read_text())
        result["token_identity"][2] = False
        result["status"] = "FAIL"
        dump(self.run / "RESULT.json", result)
        with self.assertRaises(AssertionError):
            prep.server_identity(self.run, self.configured, self.server)

    def test_rejects_run_of_another_worker(self):
        configured = copy.deepcopy(self.configured)
        configured["worker_environment"].pop("S43_PIXEL_CPU_POLL")
        with self.assertRaises(AssertionError):
            prep.server_identity(self.run, configured, self.server)

    def test_rejects_run_on_another_server(self):
        self.server.write_bytes(b"rebuilt")
        with self.assertRaises(AssertionError) as caught:
            prep.server_identity(self.run, self.configured, self.server)
        self.assertIn("another server", str(caught.exception))

    def test_idle_stop_must_launch_the_configured_worker(self):
        c = self.configured
        idle = {"status": "PASS", "preflight": {"observed_sha256_by_path": c["expected_sha256_by_path"]},
                "launch": {"worker_pids": [1], "command": ["adb", "shell", "-T", launch(
                    c["worker_path"], c["library_directories"][0], c["shard_path"], c["worker_environment"])]},
                "connections": [{}, {}],
                "stop": {"signalled": True, "boot_unchanged": True, "forward_removed": True,
                         "worker_pids_after": [], "exit_code": 0}}
        dump(self.root / "idle/RESULT.json", idle)
        self.assertTrue(prep.idle_lifecycle(self.root / "idle", c).startswith("sha256:"))
        reference = config()["helper_phone_qualified_reference"]
        with self.assertRaises(AssertionError):
            prep.idle_lifecycle(self.root / "idle", reference)


@unittest.skipUnless((CPUGPU / "physical/tcp1/SUITE_RESULT.json").is_file(), "real evidence copies absent")
class RealEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.evidence = Path(self.temporary.name)
        for name in ("tcp1", "d1", "d3"):
            (self.evidence / name).symlink_to(CPUGPU / "physical" / name)
        (self.evidence / "suites").symlink_to(CPUGPU / "suites")
        gate = config()
        self.configured, self.reference = gate["helper_phone"], gate["helper_phone_qualified_reference"]

    def tearDown(self):
        self.temporary.cleanup()

    def test_real_fixed_worker_evidence_bridges_to_the_qualified_worker(self):
        qualified = {value[7:] for value in self.reference["expected_sha256_by_path"].values()}
        bridge = prep.byte_identity_bridge(self.evidence, self.configured, self.reference, qualified)
        self.assertEqual(bridge["tcp_rows_1_2_4"]["candidate_arms"], ["c-boost-batch"])
        self.assertEqual(bridge["tcp_rows_1_2_4"]["reference_arms"], ["a-prod", "d-prod"])
        self.assertEqual(bridge["phone_local_rows_3"]["candidate_arms"], ["b-boost-batch"])
        self.assertEqual(len(bridge["phone_local_flag_matrix"]["identical_cpu_arms"]), 6)

    def test_real_calibration_reflects_the_fixed_worker(self):
        kernel = prep.fixed_worker_kernel(self.evidence / "tcp1", self.configured)
        self.assertEqual(kernel["arms"], ["c-boost-batch"])
        self.assertLess(kernel["by_rows"][1]["compute_mean_us"], 10_000)
        self.assertLess(kernel["by_rows"][4]["compute_mean_us"], 25_000)
        self.assertGreater(kernel["effective_logical_bytes_per_s"], 50 * 10**9)
        self.assertGreater(kernel["effective_logical_ops_per_s"], kernel["effective_logical_bytes_per_s"])


if __name__ == "__main__":
    unittest.main()
