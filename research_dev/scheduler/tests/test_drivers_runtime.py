"""Portable drivers, launching parts: the host driver, the direct-USB phone driver, the meters."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.configuration.site import SiteManifest  # noqa: E402
from research_dev.scheduler.drivers.adb import AdbGuard, AdbUnavailable  # noqa: E402
from research_dev.scheduler.drivers.host import (  # noqa: E402
    HostLaunchPolicy,
    LinuxNvidiaHostDriver,
    ServerLoadFailure,
    shared_layer_environment,
)
from research_dev.scheduler.drivers.meter import (  # noqa: E402
    AssumedPhoneMeter,
    PhoneSysfsMeter,
    parse_phone_power,
    phone_meter,
)
from research_dev.scheduler.drivers.phone import HelperBinding, HelperShard, phone_driver  # noqa: E402
from research_dev.scheduler.drivers.phone_functionfs import (  # noqa: E402
    FunctionFsPhoneDriver,
    session_manifest_rows,
)

from test_drivers_base import Recorder  # noqa: E402
from test_site_manifest import site_document  # noqa: E402

ARTIFACT = "sha256:" + "a" * 64
BUNDLE = "/data/local/tmp/bundle-a"


def site(directory=None) -> SiteManifest:
    document = site_document()
    document["phones"][0]["functionfs"] = {
        "session_script": "/data/local/tmp/unified/session.sh", "session_script_sha256": "sha256:" + "1" * 64,
        "restore_script": "/data/local/tmp/unified/restore.sh", "restore_script_sha256": "sha256:" + "2" * 64,
        "session_root": "/data/local/tmp/site-a/sessions", "android_gadget_path": "/config/usb_gadget/g1",
        "functionfs_gadget_path": "/config/usb_gadget/g2", "functionfs_root_path": "/dev/usb-ffs/s41",
        "phone_usb_controller": "a600000.dwc3", "kernel_release": "6.12-a", "bridge_path": "/repo/bin/usb-bridge",
        "bridge_sha256": "sha256:" + "3" * 64, "bridge_port": 27391, "minimum_usb_speed_mbps": 5000}
    if directory is not None:
        document["paths"].update(work_root=str(directory), server_path=str(Path(directory) / "bin" / "llama-server"))
    return SiteManifest.from_json(document, base=Path("/base"))


def binding(sessions=2, columns=3200) -> HelperBinding:
    shards = tuple(HelperShard(f"S{index}", f"/data/local/tmp/site-a/shards/S{index}.ffn.gguf",
                               "sha256:" + str(index + 4) * 64, tuple(range(index * 2, index * 2 + 2)), columns, 1000)
                   for index in range(sessions))
    return HelperBinding(ARTIFACT, 64, 6400, "swiglu", shards,
                         {BUNDLE + "/llama-ffn-split-worker": "sha256:" + "9" * 64})


class HostPolicyTests(unittest.TestCase):
    def test_arguments_and_environment_of_a_shared_launch(self):
        policy = HostLaunchPolicy(gpu_layers=47, ctx_size=8192, parallel=4, host_share="dormant")
        driver = LinuxNvidiaHostDriver(site())
        link = {"S41_SERVER_FFN_HOST": "127.0.0.1", "S41_SERVER_FFN_PORT": "27391"}
        argv, environment = driver.command("model-a", "alias-a", 27431, policy, binding=binding(),
                                           helper_environment=link, output=Path("/work/run"))
        self.assertEqual(argv, [
            "/repo/bin/llama-server", "-m", "/models/model-a.gguf", "--alias", "alias-a", "--host", "127.0.0.1",
            "--port", "27431", "-c", "8192", "-b", "512", "-ub", "512", "-ngl", "47", "-t", "32", "-tb", "32",
            "--parallel", "4", "--flash-attn", "off", "--no-warmup", "--kv-unified", "--cache-ram", "0",
            "--slot-save-path", "/work/run/slot-save"])
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "1")         # the site's serving board, by index
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/repo/bin")
        self.assertEqual({key: value for key, value in environment.items() if key.startswith("S41_")}, {
            "S41_SERVER_FFN_ACTIVATION": "swiglu", "S41_SERVER_FFN_ARTIFACT_SHA256": ARTIFACT,
            "S41_SERVER_FFN_COLUMNS": "3200", "S41_SERVER_FFN_F16_IO": "1", "S41_SERVER_FFN_MAX_TOKENS": "4",
            "S41_SERVER_FFN_N_EMBD": "64", "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
            "S41_SERVER_FFN_TIMEOUT_MS": "120000", "S41_SERVER_FFN_DORMANT_HOST_SHARE": "1",
            "S41_SERVER_FFN_DORMANT_DROP_CACHE": "0", "S41_SERVER_FFN_DORMANT_POPULATE": "1", **link})

    def test_resident_share_alone_launch_and_refusals(self):
        policy = HostLaunchPolicy(gpu_layers=47, ctx_size=4096, parallel=1)
        resident = shared_layer_environment(binding(), policy)
        self.assertEqual(resident["S41_SERVER_FFN_DORMANT_HOST_SHARE"], "0")
        self.assertNotIn("S41_SERVER_FFN_DORMANT_POPULATE", resident)
        driver = LinuxNvidiaHostDriver(site())
        argv, environment = driver.command("model-a", "a", 27431, policy, binding=None, helper_environment={},
                                           output=Path("/work/run"))
        self.assertFalse(any(key.startswith("S41_") for key in environment))      # alone: no split variable
        for arguments, message in (
                (("model-x", None, {}), "no model"),
                (("model-a", binding(), {}), "both the helper binding and its link"),
                (("model-a", None, {"S41_SERVER_FFN_PORT": "1"}), "both the helper binding and its link"),
                (("model-a", binding(), {"LD_PRELOAD": "x"}), "helper link environment sets LD_PRELOAD")):
            with self.assertRaisesRegex(PhysicalAdapterError, message):
                driver.command(arguments[0], "a", 27431, policy, binding=arguments[1],
                               helper_environment=arguments[2], output=Path("/work/run"))
        for options in ({"host_share": "frozen"}, {"parallel": 0}, {"ctx_size": 0}):
            with self.assertRaises(PhysicalAdapterError):
                HostLaunchPolicy(**{"gpu_layers": 1, "ctx_size": 8, "parallel": 1, **options})
        self.assertEqual(HostLaunchPolicy.from_json(policy.to_json()), policy)

    def test_cpu_affinity_wraps_the_server(self):
        document = site_document()
        document["host"]["cpu_affinity"] = "0-31"
        driver = LinuxNvidiaHostDriver(SiteManifest.from_json(document, base=Path("/base")))
        argv, _ = driver.command("model-a", "a", 27431, HostLaunchPolicy(1, 8, 1), binding=None,
                                 helper_environment={}, output=Path("/work/run"))
        self.assertEqual(argv[:4], ["taskset", "--cpu-list", "0-31", "/repo/bin/llama-server"])


class FakeProcess:
    def __init__(self, exits=None):
        self.returncode, self.pid, self.signals, self._exits = None, 4242, [], exits

    def poll(self):
        if self._exits is not None and self.returncode is None:
            self.returncode = self._exits
        return self.returncode

    def send_signal(self, number):
        self.signals.append(number)
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class HostLaunchTests(unittest.TestCase):
    def driver(self, directory, *, process=None, health=(200,), inherited=None):
        state = {"popen": [], "requests": [], "health": list(health), "t": 0.0}

        def popen(argv, env, stdout, **_):
            state["popen"].append((argv, env))
            stdout.write("server starting\n")
            return process or FakeProcess()

        def request(port, method, path, body=None, timeout_s=30):
            state["requests"].append((method, path, body))
            if path == "/health":
                return (state["health"].pop(0) if len(state["health"]) > 1 else state["health"][0]), None
            if path == "/slots":
                return 200, [{"id": 0, "is_processing": False}, {"id": 1, "is_processing": True}]
            if path == "/tokenize":
                return 200, {"tokens": [1, 2, 3]}
            return 200, {"id_slot": 2, "n_erased": 77}

        driver = LinuxNvidiaHostDriver(
            site(directory), popen=popen, request=request, clock=lambda: state["t"],
            sleep=lambda seconds: state.__setitem__("t", state["t"] + seconds),
            environ={"PATH": "/bin", "S41_SERVER_FFN_COLUMNS": "inherited"} if inherited is None else inherited)
        return driver, state

    def test_launch_scrubs_inherited_split_variables_and_records_the_command(self):
        with tempfile.TemporaryDirectory() as directory:
            driver, state = self.driver(directory)
            output = Path(directory) / "run"
            output.mkdir()
            handle = driver.launch("model-a", "a", 28411, HostLaunchPolicy(47, 8192, 4), binding=None,
                                   helper_environment={}, output=output)
            argv, environment = state["popen"][0]
            self.assertNotIn("S41_SERVER_FFN_COLUMNS", environment)
            self.assertEqual(environment["PATH"], "/bin")
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "1")
            recorded = json.loads((output / "SERVER_COMMAND.json").read_text())
            self.assertEqual(recorded["argv"], argv)
            self.assertNotIn("PATH", recorded["environment"])
            self.assertTrue((output / "slot-save").is_dir())
            self.assertEqual((handle.endpoint, handle.parallel, handle.pid), ("http://127.0.0.1:28411", 4, 4242))
            driver.wait_ready(handle, 30)
            self.assertEqual(driver.tokenize(handle, "text"), [1, 2, 3])
            self.assertEqual(driver.slots_busy(handle), [1])
            self.assertEqual(driver.erase_slot(handle, 2)["n_erased"], 77)
            self.assertIn(("POST", "/slots/2?action=erase", {}), state["requests"])

    def test_load_failures_are_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = (("CUDA error: out of memory\n", 1, "gpu-out-of-memory"), ("segfault\n", 139, "exited"))
            for index, (text, status, kind) in enumerate(cases):
                driver, _ = self.driver(directory, process=FakeProcess(exits=status))
                output = Path(directory) / f"run{index}"
                output.mkdir()
                handle = driver.launch("model-a", "a", 28420 + index, HostLaunchPolicy(47, 8192, 4), binding=None,
                                       helper_environment={}, output=output)
                handle.log_path.write_text(text)
                with self.assertRaises(ServerLoadFailure) as raised:
                    driver.wait_ready(handle, 30)
                self.assertEqual(raised.exception.kind, kind)
            driver, _ = self.driver(directory, health=(503,))
            output = Path(directory) / "slow"
            output.mkdir()
            handle = driver.launch("model-a", "a", 28429, HostLaunchPolicy(47, 8192, 4), binding=None,
                                   helper_environment={}, output=output)
            with self.assertRaises(ServerLoadFailure) as raised:
                driver.wait_ready(handle, 5)
            self.assertEqual(raised.exception.kind, "timeout")

    def test_stop_reads_the_servers_own_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            driver, _ = self.driver(directory)
            output = Path(directory) / "run"
            output.mkdir()
            handle = driver.launch("model-a", "a", 28431, HostLaunchPolicy(47, 8192, 4), binding=None,
                                   helper_environment={}, output=output)
            handle.log_path.write_text("x\nS41SERVERFFNCALL {}\nS41SERVERFFNCALL {}\nS41SERVERFFNSHAPE {\"a\":1}\n")
            summary = driver.stop(handle)
            self.assertEqual((summary["exit_status"], summary["phone_calls"], len(summary["shape_lines"])), (0, 2, 1))
            self.assertEqual(len(handle.process.signals), 1)


class FakeSession:
    SCRIPT_FIXED_ENVIRONMENT = {}

    def __init__(self, worker, functionfs, **options):
        self.worker, self.functionfs, self.options = worker, functionfs, options
        self.active, self.launch_dropped = False, None

    def preflight(self):
        return SimpleNamespace(to_json=lambda: {"phase": "preflight"})

    def start(self, log_path):
        self.active = True
        return SimpleNamespace(to_json=lambda: {"phase": "launch", "log": str(log_path)})

    def transport_contract(self):
        return SimpleNamespace(control_port=int(self.functionfs["bridge_port"]))

    def stop(self, *, served_calls):
        self.active = False
        return SimpleNamespace(to_json=lambda: {"phase": "stop", "served_calls": served_calls})

    def release(self, *, force_close=True):
        self.active = False
        return ["gadget closed"]


class FunctionFsDriverTests(unittest.TestCase):
    def driver(self, *, listening=True, run=None, factory=FakeSession):
        the_site = site()
        adb = AdbGuard(the_site.paths.adb_path, 5037, run=run or Recorder(), listening=lambda port: listening)
        return FunctionFsPhoneDriver(the_site.phone("phone-a"), the_site.paths, adb, session_factory=factory), the_site

    def test_worker_configuration_and_session_options(self):
        driver, _ = self.driver()
        session = driver.build_session(binding())
        worker = session.worker
        self.assertEqual((worker.device_id, worker.serial, worker.backend, worker.columns, worker.column_quantum,
                          worker.max_requests, worker.as_root), ("phone-a", "SERIALA", "HTP0", 3200, 3200, 0, True))
        self.assertEqual(worker.worker_path, BUNDLE + "/llama-ffn-split-worker")
        self.assertEqual(worker.layer_mask, 0b1111)
        self.assertEqual(worker.worker_environment["ADSP_LIBRARY_PATH"], BUNDLE)
        self.assertEqual(worker.worker_environment["GGML_HEXAGON_NHVX"], "4")
        self.assertIn("/data/local/tmp/site-a/shards/S1.ffn.gguf", worker.expected_sha256_by_path)
        self.assertTrue(session.options["tolerate_launch_disconnect"])
        self.assertEqual(session.options["multi_session"]["resident_router"], BUNDLE + "/llama-ffn-split-resident-router")
        self.assertNotIn("multi_session_port_base", session.functionfs)
        self.assertIsNone(driver.build_session(binding(sessions=1)).options["multi_session"])

    def test_manifest_rows_have_twelve_fields_per_session(self):
        rows = session_manifest_rows(binding(), 27511).split(";")
        self.assertEqual(len(rows), 2)
        first, second = (row.split(",") for row in rows)
        self.assertEqual(len(first), 12)
        self.assertEqual(first[:7], ["S0", "S0", ARTIFACT, "/data/local/tmp/site-a/shards/S0.ffn.gguf", "0-1", "3200",
                                     "27511"])
        self.assertEqual((second[4], second[6], second[8], second[11]), ("2-3", "27512", "1000", "1"))
        self.assertEqual(first[10], second[10])                       # one plan id for the whole binding
        self.assertNotEqual(first[7], second[7])                      # one endpoint id per session
        self.assertEqual(session_manifest_rows(binding(), 27511), ";".join(rows))    # deterministic

    def test_start_returns_the_link_environment_and_stop_and_release_delegate(self):
        with tempfile.TemporaryDirectory() as directory:
            driver, _ = self.driver()
            endpoint = driver.start(binding(), Path(directory))
            self.assertEqual(endpoint.server_environment, {
                "S41_SERVER_FFN_HOST": "127.0.0.1", "S41_SERVER_FFN_TRANSPORT": "tcp",
                "S41_SERVER_FFN_PORT": "27391", "S41_SERVER_FFN_LAYER_MASK": "15"})
            self.assertEqual((endpoint.layers, endpoint.stored_columns, endpoint.transport),
                             ((0, 1, 2, 3), 3200, "functionfs-bridge"))
            self.assertEqual(driver.receipts["launch"]["phase"], "launch")
            self.assertTrue(driver.alive())
            with self.assertRaisesRegex(PhysicalAdapterError, "already runs a session"):
                driver.start(binding(), Path(directory))
            self.assertEqual(driver.stop(served_calls=7)["served_calls"], 7)
            self.assertFalse(driver.alive())
            self.assertEqual(driver.release(), [])
            driver.start(binding(), Path(directory))
            self.assertEqual(driver.release(), ["gadget closed"])

    def test_no_adb_server_means_no_session_and_no_adb_command(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Recorder()
            driver, _ = self.driver(listening=False, run=run)
            with self.assertRaises(AdbUnavailable):
                driver.start(binding(), Path(directory))
            with self.assertRaises(AdbUnavailable):
                driver._guarded_run(["/opt/adb", "-P", "5037", "-s", "SERIALA", "shell", "id"])
            self.assertEqual(driver._adb_state()[0], 1)
            self.assertEqual(run.commands, [])
            driver._run = Recorder()
            driver._guarded_run(["/repo/bin/usb-close", "x"])           # host tools are not adb
            self.assertEqual(len(driver._run.commands), 1)

    def test_verify_binding_refusals_and_registry(self):
        pins = {BUNDLE + "/llama-ffn-split-worker": "9" * 64, "/data/local/tmp/site-a/shards/S0.ffn.gguf": "4" * 64,
                "/data/local/tmp/site-a/shards/S1.ffn.gguf": "5" * 64}
        good = Recorder({"sha256sum": "".join(f"{digest}  {path}\n" for path, digest in pins.items())})
        driver, the_site = self.driver(run=good)
        self.assertEqual(len(driver.verify(binding())), 3)
        self.assertTrue(good.commands[0][-1].startswith("su -c"))
        bad, _ = self.driver(run=Recorder({"sha256sum": f"{'0' * 64}  {BUNDLE}/llama-ffn-split-worker\n"}))
        with self.assertRaisesRegex(PhysicalAdapterError, "differ from their pinned sha256"):
            bad.verify(binding())
        with self.assertRaisesRegex(PhysicalAdapterError, "does not pin the worker"):
            driver.worker_configuration(HelperBinding(ARTIFACT, 64, 6400, "swiglu", binding().shards))
        with self.assertRaisesRegex(PhysicalAdapterError, "column quantum"):
            driver.worker_configuration(binding(columns=1600))
        with self.assertRaisesRegex(PhysicalAdapterError, "more sessions"):
            driver.multi_session(binding(sessions=4))
        self.assertIsInstance(phone_driver(the_site, "phone-a"), FunctionFsPhoneDriver)
        with self.assertRaisesRegex(PhysicalAdapterError, "not on the direct USB transport"):
            FunctionFsPhoneDriver(the_site.phone("phone-b"), the_site.paths, driver.adb)
        self.assertEqual(driver.power_calibration().gauge_ratio, 0.42)


class PhoneMeterTests(unittest.TestCase):
    def test_sampler_lines_get_the_run_directorys_field_names(self):
        the_site = site()
        meter = PhoneSysfsMeter(AdbGuard(Path("/opt/adb"), 5037, run=Recorder(), listening=lambda port: True),
                                the_site.phone("phone-a"), Path("/repo/sampler.sh"), cpu_policies=("policy0", "policy6"))
        text = ("# start uptime 100.00 pid 5 period_s 1 nodes usb/current_now\n"
                "100.00 usb/current_now=494000 usb/voltage_now=4983000 battery/current_now=37 "
                "battery/voltage_now=4108000 battery/temp=285 "
                "/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq=3628800 other/node=5\n"
                "# stop uptime 101.00 samples 1\n")
        lines = meter.normalise(text)
        self.assertEqual(lines[1], "100.00 usb_ua=494000 usb_uv=4983000 bat_ma=37 bat_uv=4108000 bat_dc=285 m0=3628800")
        self.assertEqual(parse_phone_power("\n".join(lines))[0]["bat_dc"], 285.0)
        self.assertEqual((lines[0][0], lines[2][0]), ("#", "#"))
        self.assertEqual((meter.calibration.gauge_ratio, meter.quality), (0.42, "measured"))

    def test_start_pairs_the_clocks_and_collect_writes_the_run_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sampler = root / "sampler.sh"
            sampler.write_text("# sampler\n")
            from research_dev.scheduler.drivers.adb import sha256_file
            remote = "/data/local/tmp/site-a/sampler/phone_power_sampler.sh"
            run = Recorder({"sha256sum": f"{sha256_file(sampler)[7:]}  {remote}\n", "/proc/uptime": "5000.25 4000.0\n",
                            "cat /data/local/tmp/site-a/sampler/run-1.txt":
                                "5000.00 usb/current_now=400000 usb/voltage_now=5000000 battery/current_now=0 "
                                "battery/voltage_now=4000000\n"})
            ticks = iter((1_000, 3_000))
            meter = phone_meter(AdbGuard(Path("/opt/adb"), 5037, run=run, listening=lambda port: True),
                                site().phone("phone-a"), sampler, clock_ns=lambda: next(ticks))
            pairing = meter.start(root, "run-1")
            self.assertEqual(pairing, {"phone_uptime_s": 5000.25, "host_monotonic_ns": 2_000, "pairing_width_ns": 2_000})
            self.assertEqual(json.loads((root / "PHONE_CLOCK.json").read_text()), pairing)
            launch = next(command[-1] for command in run.commands if "nohup sh" in command[-1])
            self.assertTrue(launch.startswith("su -c"))                    # this phone's nodes need root
            self.assertIn("run-1.txt", launch)
            self.assertIn("usb/current_now usb/voltage_now battery/current_now battery/voltage_now battery/temp", launch)
            self.assertFalse(any("push" in command for command in run.commands))   # the sampler was already there
            self.assertEqual(meter.collect(root), 1)
            self.assertIn("usb_ua=400000", (root / "phone-power.txt").read_text())
            power = json.loads((root / "PHONE_POWER.json").read_text())
            self.assertEqual((power["quality"], power["calibration"]["gauge_ratio"]), ("measured", 0.42))
            self.assertTrue(any(command[-1] == "wait-for-device" for command in run.commands))
            with self.assertRaisesRegex(PhysicalAdapterError, "not started"):
                meter.collect(root)

    def test_assumed_and_absent_meters(self):
        with tempfile.TemporaryDirectory() as directory:
            the_site = site()
            assumed = phone_meter(None, the_site.phone("phone-b"), Path("/x"))
            self.assertIsInstance(assumed, AssumedPhoneMeter)
            assumed.start(Path(directory), "run")
            self.assertEqual(assumed.collect(Path(directory)), 0)
            self.assertEqual(json.loads((Path(directory) / "PHONE_POWER.json").read_text()),
                             {"quality": "assumed", "assumed_active_w": 4.5, "assumed_idle_w": 0.875})
            document = site_document()
            document["phones"][1].pop("power")
            bare = SiteManifest.from_json(document, base=Path("/base")).phone("phone-b")
            self.assertIsNone(phone_meter(None, bare, Path("/x")))
            with self.assertRaisesRegex(PhysicalAdapterError, "no measured power"):
                PhoneSysfsMeter(None, the_site.phone("phone-b"), Path("/x"))


if __name__ == "__main__":
    unittest.main()
