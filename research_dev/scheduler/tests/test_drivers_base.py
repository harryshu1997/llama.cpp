"""Portable drivers, read-only parts: the adb guard, host gates, discovery, phone health, the doctor."""
from __future__ import annotations

import subprocess
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
from research_dev.scheduler.drivers import discover, doctor, gates  # noqa: E402
from research_dev.scheduler.drivers.adb import AdbGuard, AdbUnavailable, sha256_file, tcp_listening  # noqa: E402
from research_dev.scheduler.drivers.phone import (  # noqa: E402
    HelperBinding,
    HelperShard,
    parse_phone_facts,
    parse_phone_health,
    phone_driver,
    read_phone_health,
    register_phone_driver,
)

from test_site_manifest import site_document  # noqa: E402

AWAKE = """MemTotal:       15475004 kB
MemAvailable:   10387580 kB
  Thermal Status: 0
  AC powered: false
  USB powered: true
  Wireless powered: false
  status: 2
  level: 80
  temperature: 239
  mWakefulness=Awake
S43_CPU policy0 3628800 3628800
S43_CPU policy6 4608000 4608000
S43_DATA /dev/block/dm-58 229731364 193432172 36299192 85% /data
"""
ASLEEP = AWAKE.replace("=Awake", "=Asleep").replace("policy0 3628800", "policy0 1996800") \
    .replace("policy6 4608000 ", "policy6 2438400 ")
PROCESSES = "  PID NAME\n    1 init\n 4242 com.android.systemui\n"


class Recorder:
    """A fake ``subprocess.run`` that answers by the last word(s) of the command."""

    def __init__(self, answers=None):
        self.commands, self.answers = [], answers or {}

    def __call__(self, command, **_):
        self.commands.append(list(command))
        text = " ".join(command)
        for needle, reply in self.answers.items():
            if needle in text:
                if isinstance(reply, Exception):
                    raise reply
                return SimpleNamespace(stdout=reply, stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="", returncode=0)


def guard(run, *, listening=True):
    return AdbGuard(Path("/opt/adb"), 5037, run=run, listening=lambda port: listening)


class AdbGuardTests(unittest.TestCase):
    def test_no_listener_means_no_command_at_all(self):
        run = Recorder()
        adb = guard(run, listening=False)
        self.assertFalse(adb.available())
        for call in (lambda: adb.devices(), lambda: adb.shell("SERIALA", "uname -r"),
                     lambda: adb.run("SERIALA", "push", "a", "b"), lambda: adb.remote_sha256("SERIALA", ["/x"])):
            with self.assertRaises(AdbUnavailable):
                call()
        self.assertEqual(run.commands, [])

    def test_server_control_and_mode_commands_are_never_run(self):
        run = Recorder()
        adb = guard(run)
        for command in ("start-server", "kill-server", "root", "usb", "tcpip", "reboot", "reconnect", "install"):
            with self.assertRaisesRegex(PhysicalAdapterError, "not run by the drivers"):
                adb.run("SERIALA", command)
        with self.assertRaises(PhysicalAdapterError):
            adb.run("", "shell", "id")
        self.assertEqual(run.commands, [])

    def test_calls_name_port_and_serial_and_root_is_explicit(self):
        run = Recorder({"devices": "List of devices attached\nSERIALA\tdevice\nSERIALB\tunauthorized\n\n"})
        adb = guard(run)
        self.assertEqual(adb.devices(), [("SERIALA", "device"), ("SERIALB", "unauthorized")])
        adb.shell("SERIALA", "cat /proc/meminfo")
        adb.shell("SERIALA", "ps -A", root=True)
        self.assertEqual(run.commands[1], ["/opt/adb", "-P", "5037", "-s", "SERIALA", "shell", "cat /proc/meminfo"])
        self.assertEqual(run.commands[2][-1], "su -c 'ps -A'")

    def test_push_only_into_our_directories_and_only_when_needed(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "shard.gguf"
            local.write_bytes(b"shard")
            digest = sha256_file(local)
            remote = "/data/local/tmp/site-a/shards/shard.gguf"
            present = Recorder({"sha256sum": f"{digest[7:]}  {remote}\n"})
            self.assertEqual(guard(present).push_verified("SERIALA", local, remote, digest,
                                                          allowed_roots=("/data/local/tmp/site-a",)), "present")
            self.assertFalse(any("push" in command for command in present.commands))
            state = {"pushed": False}

            def run(command, **_):
                if "push" in command:
                    state["pushed"] = True
                reply = f"{digest[7:]}  {remote}\n" if state["pushed"] and "sha256sum" in command[-1] else ""
                return SimpleNamespace(stdout=reply, stderr="", returncode=0)

            self.assertEqual(guard(run).push_verified("SERIALA", local, remote, digest,
                                                      allowed_roots=("/data/local/tmp/site-a",)), "pushed")
            for outside in ("/data/local/tmp/other/shard.gguf", "/data/local/tmp/site-a-evil/x"):
                with self.assertRaisesRegex(PhysicalAdapterError, "outside this site's directories"):
                    guard(Recorder()).push_verified("SERIALA", local, outside, digest,
                                                    allowed_roots=("/data/local/tmp/site-a",))
            with self.assertRaisesRegex(PhysicalAdapterError, "differs from its pinned sha256"):
                guard(Recorder()).push_verified("SERIALA", local, remote, "sha256:" + "0" * 64,
                                                allowed_roots=("/data/local/tmp/site-a",))
            with self.assertRaisesRegex(PhysicalAdapterError, "after the push"):
                guard(Recorder()).push_verified("SERIALA", local, remote, digest,
                                                allowed_roots=("/data/local/tmp/site-a",))

    def test_listening_reads_the_socket_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "tcp"
            table.write_text("  sl  local_address rem_address   st\n"
                             "   0: 0100007F:13AD 00000000:0000 0A 0 0\n"
                             "   1: 0100007F:6B4B 0100007F:9C40 01 0 0\n")
            self.assertTrue(tcp_listening(5037, (str(table),)))
            self.assertFalse(tcp_listening(27467, (str(table),)))       # connected, not listening
            self.assertFalse(tcp_listening(5037, (str(Path(directory) / "absent"),)))


class ForeignLoadTests(unittest.TestCase):
    def watch(self, foreign_cores, *, limit=6.0, window_s=15.0):
        """A watch on a fake host: each sample period adds ``foreign_cores[i]`` cores of other users' work
        and two cores of the run's own."""
        state = {"t": 0.0, "busy": 0, "own": 0, "i": 0, "trips": 0}

        def advance():
            cores = foreign_cores[min(state["i"], len(foreign_cores) - 1)]
            state["i"] += 1
            state["t"] += 5.0
            state["own"] += 2 * 5 * 100
            state["busy"] += int((cores + 2) * 5 * 100)

        watch = gates.ForeignLoadWatch(lambda: (1,), limit_cores=limit, window_s=window_s,
                                       on_trip=lambda: state.__setitem__("trips", state["trips"] + 1),
                                       busy=lambda: state["busy"], own=lambda pids: state["own"],
                                       clock=lambda: state["t"], clock_ns=lambda: int(state["t"] * 1e9),
                                       ticks_per_s=100)
        watch.sample()
        for _ in foreign_cores:
            advance()
            watch.sample()
        return watch, state

    def test_own_work_is_not_foreign(self):
        watch, state = self.watch([0.0, 0.0, 0.0, 0.0])
        self.assertEqual([value for _, value in watch.samples], [0.0, 0.0, 0.0, 0.0])
        self.assertTrue(watch.quiet(4.0, 15.0))
        self.assertIsNone(watch.tripped)

    def test_sustained_foreign_load_trips_once_and_rearms(self):
        watch, state = self.watch([0.0, 90.0, 90.0, 90.0, 90.0])
        self.assertEqual(state["trips"], 1)
        self.assertEqual(watch.tripped["cores"], [90.0, 90.0, 90.0])
        self.assertFalse(watch.quiet(4.0, 15.0))
        watch.rearm()
        self.assertIsNone(watch.tripped)

    def test_a_short_burst_does_not_trip_but_counts_as_disturbance(self):
        watch, state = self.watch([0.0, 40.0, 0.0, 0.0, 0.0])
        self.assertEqual(state["trips"], 0)
        seen = watch.disturbance(0, 10 ** 12, 4.0)
        self.assertEqual((seen["samples"], seen["max_cores"]), (5, 40.0))
        self.assertAlmostEqual(seen["mean_excess_cores"], 36.0 / 5)
        self.assertEqual(watch.summary(0, 10 ** 12)["samples_above_4"], 1)
        self.assertEqual(watch.summary(10 ** 13, 10 ** 14), {"samples": 0})
        self.assertTrue(watch.quiet(4.0, 15.0))                      # the last three samples are quiet
        self.assertFalse(watch.quiet(4.0, 25.0))                     # the burst is inside 25 s

    def test_gpu_state_and_wait_until(self):
        run = Recorder({"--query-compute-apps": "4711\n", "memory.used": "49000\n"})
        state = gates.gpu_free(1, run=run)
        self.assertEqual((state["free"], state["compute_pids"], state["memory_used_mib"]), (False, [4711], 49000))
        self.assertTrue(all(command[1:3] == ["-i", "1"] for command in run.commands))
        self.assertTrue(gates.gpu_free(1, run=Recorder({"memory.used": "58\n"}))["free"])
        clock = {"t": 0.0}
        tick = lambda seconds: clock.__setitem__("t", clock["t"] + seconds)  # noqa: E731
        self.assertEqual(gates.wait_until(lambda: clock["t"] >= 3, timeout_s=10, clock=lambda: clock["t"],
                                          sleep=tick), 3.0)
        with self.assertRaises(TimeoutError):
            gates.wait_until(lambda: False, timeout_s=5, clock=lambda: clock["t"], sleep=tick)
        with self.assertRaisesRegex(RuntimeError, "exited"):
            gates.wait_until(lambda: False, timeout_s=5, clock=lambda: clock["t"], sleep=tick, alive=lambda: False)


def usb_tree(root: Path, devices) -> Path:
    for name, serial, speed in devices:
        directory = root / "devices" / "pci0000:00" / "0000:00:14.0" / ("usb" + name.split("-")[0]) / name
        directory.mkdir(parents=True)
        for field, value in (("serial", serial), ("idVendor", "22d9"), ("idProduct", "2769"), ("speed", str(speed))):
            (directory / field).write_text(value + "\n")
        bus = root / "bus"
        bus.mkdir(exist_ok=True)
        (bus / name).symlink_to(directory)
    return root / "bus"


class DiscoveryTests(unittest.TestCase):
    def test_host_facts_from_a_two_gpu_listing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cpuinfo").write_text("processor\t: 0\nmodel name\t: Synthetic CPU 64\n")
            (root / "meminfo").write_text("MemTotal:       1048576 kB\nMemFree: 5 kB\n")
            for zone, name, readable in (("intel-rapl:0", "package-0", True), ("intel-rapl:1", "package-1", False),
                                         ("intel-rapl:0:0", "core", True)):
                (root / zone).mkdir()
                (root / zone / "name").write_text(name + "\n")
                if readable:
                    (root / zone / "energy_uj").write_text("12345\n")
            run = Recorder({"--query-gpu=index": "0, GPU-0000, Synthetic GPU, 49140\n1, GPU-1111, Synthetic GPU, 49140\n"})
            facts = discover.host_facts(run=run, cpuinfo=root / "cpuinfo", meminfo=root / "meminfo", rapl_root=root,
                                        cpu_count=lambda: 64)
        self.assertEqual((facts["cpu_model"], facts["logical_cpus"], facts["memory_total_bytes"]),
                         ("Synthetic CPU 64", 64, 1 << 30))
        self.assertEqual([gpu["uuid"] for gpu in facts["gpus"]], ["GPU-0000", "GPU-1111"])
        self.assertEqual(facts["gpus"][1]["memory_total_bytes"], 49140 << 20)
        self.assertEqual(facts["rapl_zones"], ["intel-rapl:0"])       # the unreadable package is left out

    def test_attached_phones_get_host_side_facts_only(self):
        with tempfile.TemporaryDirectory() as directory:
            bus = usb_tree(Path(directory), (("8-3", "SERIALA", 5000), ("5-4", "SERIALB", 480)))
            run = Recorder({"devices": "List of devices attached\nSERIALA\tdevice\nSERIALB\tdevice\nSERIALC\toffline\n"})
            rows = discover.attached_phones(guard(run), sysfs_root=bus)
            self.assertEqual(rows[0], {"serial": "SERIALA", "state": "device", "usb_sysfs_device": "8-3",
                                       "usb_speed_mbps": 5000})
            self.assertEqual(rows[1]["usb_speed_mbps"], 480)
            self.assertIn("no unique USB device", rows[2]["usb"])
            self.assertEqual(len(run.commands), 1)                    # one `adb devices`, no phone command
            facts_run = Recorder({"getprop": "S43_MODEL Model B\nS43_SOC SM8650\nS43_PLATFORM pineapple\n"
                                             "S43_KERNEL 6.1-b\nMemTotal:       11534336 kB\n", "su -c id": ""})
            facts = discover.phone_facts(guard(facts_run), "SERIALB", sysfs_root=bus)
        self.assertEqual(facts, {"model": "Model B", "soc": "SM8650", "kernel_release": "6.1-b",
                                 "memory_total_bytes": 11534336 * 1024, "root": False, "usb_sysfs_device": "5-4",
                                 "usb_speed_mbps": 480})

    def test_phone_facts_parse_root_and_fall_back_to_the_platform(self):
        text = "S43_MODEL Model C\nS43_SOC \nS43_PLATFORM kalama\nS43_KERNEL 5.15-c\nMemTotal: 8 kB\n"
        facts = parse_phone_facts(text, "uid=0(root) gid=0(root)\n")
        self.assertEqual((facts["soc"], facts["root"]), ("kalama", True))
        with self.assertRaisesRegex(PhysicalAdapterError, "did not report kernel_release,model"):
            parse_phone_facts("S43_SOC SM1\nMemTotal: 8 kB\n", "")


class PhoneHealthTests(unittest.TestCase):
    def test_awake_idle_phone(self):
        health = parse_phone_health(AWAKE, PROCESSES)
        self.assertEqual((health.awake, health.capped, health.thermal_status, health.battery_decic,
                          health.battery_level, health.charging), (True, False, 0, 239, 80, True))
        self.assertEqual(health.memory_available_bytes, 10387580 * 1024)
        self.assertEqual(health.data_free_bytes, 36299192 * 1024)
        self.assertEqual(health.workload_processes, ())
        self.assertEqual(parse_phone_health(health_text := AWAKE, PROCESSES).to_json()["cpu_caps_khz"],
                         {"policy0": 3628800, "policy6": 4608000})
        self.assertTrue(health_text)

    def test_asleep_phone_is_capped_and_workloads_are_seen(self):
        health = parse_phone_health(ASLEEP, PROCESSES + " 5150 llama-server\n 5151 /data/x/ffn-stream-client\n"
                                    " 5152 llama-ffn-split-worker\n 5153 com.example.llamafarm\n")
        self.assertEqual((health.awake, health.capped), (False, True))
        self.assertEqual(health.cpu_caps_khz["policy0"], 1996800)
        self.assertEqual(health.workload_processes, ("5150 llama-server", "5151 /data/x/ffn-stream-client",
                                                     "5152 llama-ffn-split-worker"))

    def test_missing_parts_are_none_and_root_is_used_only_for_the_process_list(self):
        health = parse_phone_health("", "")
        self.assertEqual((health.awake, health.thermal_status, health.memory_available_bytes, health.charging,
                          health.capped), (None, None, None, None, False))
        run = Recorder({"S43_CPU": AWAKE, "ps -A": PROCESSES})
        phone = SimpleNamespace(serial="SERIALA", root=True)
        read_phone_health(guard(run), phone)
        self.assertFalse(run.commands[0][-1].startswith("su -c"))
        self.assertTrue(run.commands[1][-1].startswith("su -c"))
        read_phone_health(guard(run), SimpleNamespace(serial="SERIALB", root=False))
        self.assertFalse(any(command[-1].startswith("su -c") for command in run.commands[2:]))


class HelperBindingTests(unittest.TestCase):
    def shard(self, session, layers, columns=3200):
        return HelperShard(session, f"/data/local/tmp/site-a/{session}.gguf", "sha256:" + "1" * 64, layers, columns, 10)

    def test_binding_geometry_and_round_trip(self):
        binding = HelperBinding("sha256:" + "a" * 64, 64, 6400, "swiglu",
                                (self.shard("S0", (0, 1)), self.shard("S1", (2, 3))), {"/x": "sha256:" + "2" * 64})
        self.assertEqual((binding.layers, binding.stored_columns, binding.layer_mask), ((0, 1, 2, 3), 3200, 15))
        self.assertEqual(HelperBinding.from_json(binding.to_json()), binding)

    def test_inconsistent_bindings_are_refused(self):
        for shards, activation, message in (
                ((), "swiglu", "no shard"),
                ((self.shard("S0", (0,)), self.shard("S1", (1,), 1600)), "swiglu", "same columns"),
                ((self.shard("S0", (0, 1)), self.shard("S1", (1,))), "swiglu", "distinct layers"),
                ((self.shard("S0", (0,)),), "relu", "activation")):
            with self.assertRaisesRegex(PhysicalAdapterError, message):
                HelperBinding("sha256:" + "a" * 64, 64, 6400, activation, shards)

    def test_driver_registry_is_keyed_by_transport(self):
        site = SiteManifest.from_json(site_document(), base=Path("/base"))
        from research_dev.scheduler.drivers import phone as phone_module
        made = []
        before = dict(phone_module._DRIVERS)          # other test modules register the real drivers
        try:
            register_phone_driver("adb-tcp", lambda phone, paths, adb: made.append((phone.device_id, adb.port)) or "driver")
            self.assertEqual(phone_driver(site, "phone-b"), "driver")
            self.assertEqual(made, [("phone-b", 5037)])
            phone_module._DRIVERS.clear()
            with self.assertRaisesRegex(PhysicalAdapterError, "no portable driver for transport adb-tcp"):
                phone_driver(site, "phone-b")
        finally:
            phone_module._DRIVERS.clear()
            phone_module._DRIVERS.update(before)


class DoctorTests(unittest.TestCase):
    def site(self, directory, **paths):
        document = site_document()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for name in ("server", "adb", "model", "bridge"):
            (root / name).write_text("x")
        document["paths"].update(server_path=str(root / "server"), adb_path=str(root / "adb"), work_root=str(root / "work"),
                                 models={"model-a": str(root / "model")}, host_tools={"usb_bridge": str(root / "bridge")},
                                 **paths)
        return SiteManifest.from_json(document, base=root)

    def host(self, site, directory, *, used="58\n", apps="", gpus=None, rapl=True, listening=True, busy=(0, 0)):
        rapl_root = Path(directory) / "rapl"
        (rapl_root / "intel-rapl:0").mkdir(parents=True, exist_ok=True)
        if rapl:
            (rapl_root / "intel-rapl:0" / "energy_uj").write_text("5\n")
        run = Recorder({"--query-gpu=index": gpus or "0, GPU-0000, Synthetic GPU, 49140\n1, GPU-1111, Synthetic GPU, 49140\n",
                        "--query-compute-apps": apps, "memory.used": used})
        values = iter(busy)
        checks = doctor.host_checks(site, run=run, rapl_root=rapl_root, listening=lambda port: listening,
                                    sleep=lambda seconds: None, busy=lambda: next(values))
        return {check.check_id: check for check in checks}

    def test_clean_host(self):
        with tempfile.TemporaryDirectory() as directory:
            checks = self.host(self.site(directory), directory)
        self.assertEqual({check.status for check in checks.values()}, {"ok"})
        self.assertEqual(sorted(checks), ["adb-server:5037", "host-files", "host-gpu", "host-gpu-free", "host-load",
                                          "host-rapl"])

    def test_every_host_fault_blocks_with_a_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            site = self.site(directory)
            busy_gpu = self.host(site, directory, used="48000\n", apps="4711\n")
            other_board = self.host(site, directory, gpus="0, GPU-0000, G, 49140\n1, GPU-9999, G, 49140\n")
            no_rapl = self.host(site, directory + "/b", rapl=False)
            no_adb = self.host(site, directory, listening=False)
            loaded = self.host(site, directory, busy=(0, 90 * 100 * 2))
            (Path(directory) / "model").unlink()
            no_model = self.host(site, directory)
        self.assertEqual(busy_gpu["host-gpu-free"].status, "blocking")
        self.assertIn("not ours to stop", busy_gpu["host-gpu-free"].fix)
        self.assertEqual(other_board["host-gpu"].status, "blocking")
        self.assertNotIn("host-gpu-free", other_board)
        self.assertEqual(no_rapl["host-rapl"].status, "blocking")
        self.assertIn("administrator", no_rapl["host-rapl"].fix)
        self.assertEqual(no_adb["adb-server:5037"].status, "blocking")
        self.assertIn("none is started here", no_adb["adb-server:5037"].fix)
        self.assertEqual(loaded["host-load"].status, "degraded")       # a run waits; it does not refuse
        self.assertEqual(no_model["host-files"].status, "blocking")
        self.assertEqual(len(doctor.blocking(list(no_model.values()))), 1)

    def phone(self, directory, device_id, *, state=AWAKE, processes=PROCESSES, listing=None, kernel="6.12-a",
              devices=None, speed=None, listening=False, power_ok="ok\nok\nok\nok\n", locked=False, metered=True):
        site = self.site(directory)
        phone = site.phone(device_id)
        bus = usb_tree(Path(directory) / device_id, (("8-3", "SERIALA", speed or 5000), ("5-4", "SERIALB", speed or 480)))
        listing = "llama-ffn-split-worker\nlibggml-htp-v81.so\nlibggml-htp-v75.so\n" if listing is None else listing
        # the health command names /sys/devices/...: its needle goes before the plain "devices" one
        run = Recorder({"S43_CPU": state, "ps -A": processes, "uname -r": kernel + "\n", "ls ": listing,
                        "test -r": power_ok, "devices": devices or "List\nSERIALA\tdevice\nSERIALB\tdevice\n"})
        checks = doctor.phone_checks(site, phone, guard(run), usb_root=bus, listening=lambda port: listening,
                                     lock_held=lambda root: locked, metered=metered)
        return {check.check_id.split(":")[0]: check for check in checks}, run

    def test_healthy_direct_usb_phone(self):
        with tempfile.TemporaryDirectory() as directory:
            checks, run = self.phone(directory, "phone-a")
        self.assertEqual({name: check.status for name, check in checks.items() if check.status != "ok"}, {})
        self.assertIn("libggml-htp-v81.so", checks["phone-bundle"].detail)
        self.assertFalse(any(word in " ".join(command) for command in run.commands
                             for word in ("start-server", "kill-server", "reboot")))

    def test_unrooted_usb2_phone_is_degraded_not_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            checks, run = self.phone(directory, "phone-b")
        self.assertEqual({name: check.status for name, check in checks.items() if check.status != "ok"},
                         {"phone-usb": "degraded", "phone-root": "degraded", "phone-power": "degraded"})
        self.assertIn("assumed 4.5 W", checks["phone-power"].detail)
        self.assertNotIn("phone-kernel", checks)
        self.assertFalse(any(command[-1].startswith("su -c") for command in run.commands))

    def test_sleeping_capped_phone_blocks_a_metered_run_only(self):
        with tempfile.TemporaryDirectory() as directory:
            metered, _ = self.phone(directory, "phone-a", state=ASLEEP)
            unmetered, _ = self.phone(directory + "/u", "phone-a", state=ASLEEP, metered=False)
        self.assertEqual((metered["phone-awake"].status, metered["phone-cpu"].status), ("blocking", "blocking"))
        self.assertIn("policy0 1996 MHz (needs 3628)", metered["phone-cpu"].detail)
        self.assertEqual((unmetered["phone-awake"].status, unmetered["phone-cpu"].status), ("degraded", "degraded"))

    def test_phone_faults(self):
        with tempfile.TemporaryDirectory() as directory:
            used, _ = self.phone(directory + "/1", "phone-b", processes=PROCESSES + " 5150 llama-server\n")
            kernel, _ = self.phone(directory + "/2", "phone-a", kernel="6.12-stock")
            library, _ = self.phone(directory + "/3", "phone-b", listing="llama-ffn-split-worker\nlibggml-htp-v81.so\n")
            bundle, _ = self.phone(directory + "/4", "phone-b", listing="")
            offline, _ = self.phone(directory + "/5", "phone-b", devices="List\nSERIALB\tunauthorized\n")
            ports, _ = self.phone(directory + "/6", "phone-a", listening=True)
            hot, _ = self.phone(directory + "/7", "phone-a", state=AWAKE.replace("temperature: 239", "temperature: 351"))
            nodes, _ = self.phone(directory + "/8", "phone-a", power_ok="ok\nok\n")
            locked, run = self.phone(directory + "/9", "phone-a", locked=True)
        self.assertEqual(used["phone-idle"].status, "blocking")
        self.assertIn("5150 llama-server", used["phone-idle"].detail)
        self.assertEqual(kernel["phone-kernel"].status, "blocking")
        self.assertIn("nothing here runs it", kernel["phone-kernel"].fix)
        self.assertEqual(library["phone-bundle"].status, "blocking")
        self.assertIn("SM8650 needs libggml-htp-v75.so", library["phone-bundle"].detail)
        self.assertEqual(bundle["phone-bundle"].status, "blocking")
        self.assertEqual(list(offline), ["phone-usb", "phone-adb"])
        self.assertEqual(offline["phone-adb"].status, "blocking")
        self.assertEqual(ports["phone-ports"].status, "blocking")
        self.assertEqual(hot["phone-thermal"].status, "blocking")
        self.assertEqual(nodes["phone-power"].status, "blocking")
        self.assertEqual([check.status for check in locked.values()], ["skipped"])
        self.assertEqual(run.commands, [])                               # a metered run is on: the phone is not touched

    def test_a_capped_slow_cluster_degrades_and_the_caps_floor_is_the_minimum_seen(self):
        from research_dev.scheduler.drivers.phone import parse_caps_floor, read_phone_caps_floor
        # awake, but the little cluster is capped (it moves from second to second on some phones)
        little = AWAKE.replace("policy0 3628800 3628800", "policy0 1459200 2265600")
        phone = SiteManifest.from_json(site_document(), base=Path("/base")).phone("phone-b")
        health = parse_phone_health(little, PROCESSES)
        rows = {check.check_id: check for check in doctor.health_checks(phone, health)}
        self.assertEqual(rows["phone-cpu:phone-b"].status, "degraded")
        self.assertIn("slower CPU cluster", rows["phone-cpu:phone-b"].detail)
        # with a recorded floor the phone is held to it: at the floor passes, below it blocks
        held = {check.check_id: check for check in doctor.health_checks(
            phone, health, required_caps_khz={"policy0": 1459200, "policy6": 4608000})}
        self.assertEqual(held["phone-cpu:phone-b"].status, "ok")
        below = {check.check_id: check for check in doctor.health_checks(
            phone, health, required_caps_khz={"policy0": 2265600, "policy6": 4608000})}
        self.assertEqual(below["phone-cpu:phone-b"].status, "blocking")
        text = ("S43_CAP policy0 2265600\nS43_CAP policy6 4608000\nS43_CAP policy0 1459200\n"
                "S43_CAP policy6 4608000\nS43_CAP policy0 2265600\n")
        self.assertEqual(parse_caps_floor(text), {"policy0": 1459200, "policy6": 4608000})
        run = Recorder({"S43_CAP": text})
        floor = read_phone_caps_floor(guard(run), SimpleNamespace(serial="SERIALB"), samples=3, period_s=0.5)
        self.assertEqual(floor["policy0"], 1459200)
        self.assertIn("for i in 0 1 2;", run.commands[0][-1])
        self.assertFalse(run.commands[0][-1].startswith("su -c"))

    def test_memory_need_and_recorded_caps(self):
        health = parse_phone_health(ASLEEP.replace("=Asleep", "=Awake"), PROCESSES)
        phone = SiteManifest.from_json(site_document(), base=Path("/base")).phone("phone-a")
        low = {check.check_id: check for check in doctor.health_checks(phone, health, needed_memory_bytes=12 << 30)}
        self.assertEqual(low["phone-memory:phone-a"].status, "blocking")
        # the caps recorded at onboarding are the bar, not the hardware maximum
        ok = {check.check_id: check for check in doctor.health_checks(
            phone, health, required_caps_khz={"policy0": 1996800, "policy6": 2438400})}
        self.assertEqual(ok["phone-cpu:phone-a"].status, "ok")
        self.assertIn("BLOCKING", doctor.format_checks(list(low.values())))

    def test_site_checks_hold_each_phone_to_its_recorded_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            site = self.site(directory)
            seen = {}

            def fake_phone_checks(site_, phone, adb, *, metered, required_caps_khz, **_):
                seen[phone.device_id] = required_caps_khz
                return []

            original, original_host = doctor.phone_checks, doctor.host_checks
            doctor.phone_checks, doctor.host_checks = fake_phone_checks, lambda site_: []
            try:
                doctor.site_checks(site, required_caps_by_phone={"phone-b": {"policy0": 1459200}, "phone-a": {}})
                self.assertEqual(seen, {"phone-a": None, "phone-b": {"policy0": 1459200}})
                seen.clear()
                doctor.site_checks(site, phones=["phone-b"])
                self.assertEqual(seen, {"phone-b": None})
            finally:
                doctor.phone_checks, doctor.host_checks = original, original_host

    def test_run_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(doctor.RunLock.held(root))
            with doctor.RunLock(root):
                self.assertTrue(doctor.RunLock.held(root))
                with self.assertRaisesRegex(PhysicalAdapterError, "another metered run"):
                    doctor.RunLock(root).__enter__()
            self.assertFalse(doctor.RunLock.held(root))


if __name__ == "__main__":
    unittest.main()
