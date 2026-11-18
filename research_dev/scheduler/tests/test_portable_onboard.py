"""Portable path: the site profile, onboarding from run directories, and the generic adb-tcp phone driver."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.layer_placement import (  # noqa: E402
    HelperDevice,
    LayerPlacementError,
    LayerPlacementProfile,
    ModelLayers,
    PlacementProblem,
    solve_placement,
)
from research_dev.scheduler._internal.shared_layer import PhoneCapacity, StaticSharePlan  # noqa: E402
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.campaigns.portable import onboard  # noqa: E402
from research_dev.scheduler.campaigns.portable.rundir import RunDirectory  # noqa: E402
from research_dev.scheduler.configuration.common import SchedulerConfigurationError  # noqa: E402
from research_dev.scheduler.configuration.site import SiteManifest  # noqa: E402
from research_dev.scheduler.configuration.site_profile import SiteProfile  # noqa: E402
from research_dev.scheduler.drivers.adb import AdbGuard, AdbUnavailable  # noqa: E402
from research_dev.scheduler.drivers.phone import HelperBinding, HelperShard, parse_phone_health, phone_driver  # noqa: E402
from research_dev.scheduler.drivers.phone_adb_tcp import AdbTcpPhoneDriver  # noqa: E402

import portable_fixtures as fx  # noqa: E402
from test_drivers_base import AWAKE, PROCESSES, Recorder  # noqa: E402
from test_drivers_runtime import ARTIFACT, site  # noqa: E402
from test_site_manifest import site_document  # noqa: E402

S = fx.SECOND
N_EMBD, N_FF = 64, fx.N_FF


def write_gguf(path: Path, *, architecture="qwen3", blocks=8, weight_type=1, n_embd=N_EMBD, n_ff=N_FF) -> Path:
    """A header-only GGUF: architecture, block count and the FFN tensor directory (no tensor data needed)."""
    def text(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    fields = [text("general.architecture") + struct.pack("<I", 8) + text(architecture),
              text(architecture + ".block_count") + struct.pack("<I", 4) + struct.pack("<I", blocks)]
    tensors = []
    for layer in range(blocks):
        for name, shape in (("gate", (n_embd, n_ff)), ("up", (n_embd, n_ff)), ("down", (n_ff, n_embd))):
            tensors.append(text(f"blk.{layer}.ffn_{name}.weight") + struct.pack("<I", 2)
                           + struct.pack("<QQ", *shape) + struct.pack("<I", weight_type) + struct.pack("<Q", 0))
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), len(fields))
                     + b"".join(fields) + b"".join(tensors))
    return path


def sweep_run(root: Path, *, layers=(0, 1, 2, 3), t_host=8.0, c_host=0.5, t_link=1.0, t_phone=16.0,
              other_ms=20.0, shares=(1600, 3200, 4800)) -> Path:
    """A run whose legs and shape rows follow the shared-layer model exactly (100 tokens per leg)."""
    def token_ms(columns):
        share = columns / N_FF
        layer = t_host if not columns else max(c_host + (1 - share) * t_host, t_link + share * t_phone)
        return other_ms + len(layers) * layer

    order = [0]
    for columns in shares:
        order += [columns, 0]
    phases, busy, instant = [], [], 10 * S
    for index, columns in enumerate(order):
        first = instant + S
        finished = first + int(token_ms(columns) * 100 * 1e6)
        busy.append((first, finished))
        phases.append(fx.phase(f"leg{index:02d}c{columns}", [fx.request("probe", columns, instant, first, finished, 101)]))
        instant = finished + 2 * S
    shapes = [{"tokens": 1, "columns": columns, "calls": 400,
               "overlap_mean_ms": max(c_host + (1 - columns / N_FF) * t_host, t_link + columns / N_FF * t_phone),
               "host_mean_ms": c_host + (1 - columns / N_FF) * t_host, "rpc_mean_ms": t_link + columns / N_FF * t_phone,
               "compute_mean_ms": columns / N_FF * t_phone} for columns in shares]
    return fx.write_run(root, phases=phases, layers=layers, shapes=shapes, idle=(2 * S, 8 * S, 100.0, 20.0),
                        samples=fx.power_samples(0, instant + 5 * S, lambda t: (200.0, 100.0) if any(
                            a <= t <= b for a, b in busy) else (100.0, 20.0)),
                        phone_power={"assumed_active_w": 3.0, "assumed_idle_w": 1.0})


def onboarded(directory, **sweep_options):
    """A site with a model file, a host record and one ingested sweep of phone-b."""
    root = Path(directory)
    document = site_document()
    document["paths"].update(work_root=str(root / "work"), models={"model-a": str(write_gguf(root / "model-a.gguf"))})
    document["phones"][1]["column_quantum"] = 400          # this phone's worker serves sixteenths of the layer
    the_site = SiteManifest.from_json(document, base=root)
    profile = SiteProfile(the_site.site_id)
    run = RunDirectory(sweep_run(root / "run-sweep", **sweep_options))
    onboard.ingest_host(the_site, profile, "model-a", run, gpu_layers=5, artifact_sha256=ARTIFACT,
                        memory_available=64 << 30)
    return the_site, profile, run


class ModelGeometryTests(unittest.TestCase):
    def test_geometry_and_cpu_layers_from_the_gguf(self):
        with tempfile.TemporaryDirectory() as directory:
            geometry = onboard.model_geometry(write_gguf(Path(directory) / "m.gguf"), gpu_layers=5)
            self.assertEqual((geometry["architecture"], geometry["activation"], geometry["block_count"],
                              geometry["n_embd"], geometry["n_ff"], geometry["weight_type"]),
                             ("qwen3", "swiglu", 8, N_EMBD, N_FF, "F16"))
            self.assertEqual(geometry["cpu_layers"], [0, 1, 2, 3])        # 8 blocks, -ngl 5: blocks 4-7 on the GPU
            self.assertEqual(geometry["layer_bytes"], 3 * N_EMBD * N_FF * 2)
            self.assertEqual(onboard.model_geometry(Path(directory) / "m.gguf", gpu_layers=0)["cpu_layers"],
                             list(range(8)))
            self.assertEqual(onboard.model_geometry(Path(directory) / "m.gguf", gpu_layers=99)["cpu_layers"], [])
            with self.assertRaisesRegex(PhysicalAdapterError, "no FFN split"):
                onboard.model_geometry(write_gguf(Path(directory) / "x.gguf", architecture="bert"), 5)

    def test_host_share_follows_memory(self):
        self.assertEqual(onboard.host_share(60 << 30, 113 << 30), "resident")
        self.assertEqual(onboard.host_share(60 << 30, 62 << 30), "dormant")


class OnboardingTests(unittest.TestCase):
    def test_host_record_from_host_only_legs(self):
        with tempfile.TemporaryDirectory() as directory:
            the_site, profile, run = onboarded(directory)
            host = profile.host_model("model-a")
            self.assertAlmostEqual(host["token_ms_alone"], 20.0 + 4 * 8.0, places=6)
            self.assertAlmostEqual(host["p_idle_w"], 120.0, places=6)
            self.assertAlmostEqual(host["p_decode_w"], 300.0, places=3)
            self.assertEqual((host["host_share"], host["cpu_layers"], host["legs"], host["evidence"]),
                             ("resident", [0, 1, 2, 3], 4, ["run-sweep"]))
            self.assertEqual(profile.revision, 1)

    def test_phone_sweep_gives_fit_plan_host_numbers_and_planner_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            the_site, profile, run = onboarded(directory)
            health = parse_phone_health(AWAKE, PROCESSES)
            capacity = onboard.phone_capacity(the_site, "phone-b", health.memory_available_bytes)
            self.assertEqual(capacity.usable_bytes, int(10387580 * 1024 * 0.75))
            pair = onboard.ingest_phone(the_site, profile, "phone-b", "model-a", [run], capacity=capacity,
                                        health=health)
            fit, plan = pair["fit"], StaticSharePlan.from_json(pair["share_plan"])
            self.assertAlmostEqual(fit["t_host_ms"], 8.0, places=9)
            self.assertAlmostEqual(fit["f_star"], 7.5 / 24.0, places=9)
            self.assertEqual((plan.columns, plan.layers, plan.binding_constraint), (2000, (0, 1, 2, 3), "optimum"))
            host = profile.host_model("model-a")
            self.assertAlmostEqual(host["t_host_ms"], 8.0, places=9)
            self.assertAlmostEqual(host["c_host_ms"], 0.5, places=9)
            self.assertAlmostEqual(host["other_step_ms"], 20.0, places=6)
            self.assertAlmostEqual(host["p_f_w"], 300.0, places=2)        # constant decode power: P_F = 300 W
            record = profile.phones["phone-b"]
            self.assertEqual((record["transport"], record["power_quality"], record["awake"],
                              record["cpu_caps_khz"]["policy0"]), ("adb-tcp", "assumed", True, 3628800))
            # no measured leg at the planned share (the sweep visited other shares)
            self.assertIsNone(pair["measured_at_plan_share"])
            text = onboard.format_pair(pair, host)
            self.assertIn("both end together at 31.2 %", text)
            self.assertIn("2000 columns (31.2 %) of 4 of 4 CPU layers in 1 session", text)
            # the planner prices this pair from the profile alone
            planner = LayerPlacementProfile.from_json(profile.layer_placement_profile)
            model = ModelLayers("model-a", (0, 1, 2, 3), {"f16": host["layer_bytes"]}, n_embd=N_EMBD, n_ff=N_FF)
            device = HelperDevice("phone-b", {"model-a": "f16"}, 1 << 40, "adb-tcp")
            placed = solve_placement(PlacementProblem((model,), (device,), planner,
                                                      column_fractions={"model-a": (0.25,)})).models["model-a"]
            self.assertAlmostEqual(placed.chain_ms_by_rows[1], 4 * max(0.75 * 8.0, 1.0 + 0.25 * 16.0))
            # the profile survives its own JSON
            again = SiteProfile.from_json(json.loads(json.dumps(profile.to_json())))
            self.assertEqual(again.to_json(), profile.to_json())

    def test_measured_saving_is_pooled_only_over_runs_with_the_plans_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            the_site, profile, run = onboarded(directory, shares=(1600, 2000, 3200, 4800))
            probe = RunDirectory(sweep_run(Path(directory) / "run-probe", layers=(0,), shares=(1600, 2000, 3200)))
            pair = onboard.ingest_phone(the_site, profile, "phone-b", "model-a", [probe, run],
                                        capacity=PhoneCapacity(1 << 40))
            measured = pair["measured_at_plan_share"]
            self.assertEqual((pair["share_plan"]["columns"], measured["legs"], pair["evidence"]),
                             (2000, 1, ["run-probe", "run-sweep"]))
            # 4 layers x (8.0 - 6.0) ms of a 52 ms token; the 1-layer probe run would show a quarter of it
            self.assertAlmostEqual(measured["time_saving_pct"], 8.0 / 52.0 * 100, places=3)
            self.assertFalse(measured["phone_metered"] is None)

    def test_slow_link_and_small_phone(self):
        with tempfile.TemporaryDirectory() as directory:
            the_site, profile, _ = onboarded(directory)
            slow = RunDirectory(sweep_run(Path(directory) / "run-slow", t_link=9.0))
            pair = onboard.ingest_phone(the_site, profile, "phone-b", "model-a", [slow], capacity=PhoneCapacity(1 << 40))
            self.assertEqual(pair["share_plan"]["binding_constraint"], "not-beneficial")
            self.assertIn("not beneficial", onboard.format_pair(pair, profile.host_model("model-a")))
            layer_bytes = profile.host_model("model-a")["layer_bytes"]
            small = onboard.ingest_phone(the_site, profile, "phone-b", "model-a",
                                         [RunDirectory(Path(directory) / "run-sweep")],
                                         capacity=PhoneCapacity(layer_bytes // 2))
            plan = small["share_plan"]
            self.assertEqual(plan["binding_constraint"], "memory")
            self.assertLess(len(plan["layers"]), 4)
            self.assertLessEqual(plan["resident_bytes"], layer_bytes // 2)
            self.assertEqual(len(profile.pairs), 1)                         # the pair was replaced, not added
            # a later sweep keeps the host numbers the first one wrote, unless asked to refresh them
            self.assertAlmostEqual(profile.host_model("model-a")["t_host_ms"], 8.0, places=9)
            faster = RunDirectory(sweep_run(Path(directory) / "run-faster-host", t_host=7.0))
            onboard.ingest_phone(the_site, profile, "phone-b", "model-a", [faster], capacity=PhoneCapacity(1 << 40))
            self.assertAlmostEqual(profile.host_model("model-a")["t_host_ms"], 8.0, places=9)
            self.assertAlmostEqual(profile.pair("phone-b", "model-a")["fit"]["t_host_ms"], 7.0, places=9)
            onboard.ingest_phone(the_site, profile, "phone-b", "model-a", [faster], capacity=PhoneCapacity(1 << 40),
                                 update_host=True)
            self.assertAlmostEqual(profile.host_model("model-a")["t_host_ms"], 7.0, places=9)

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            the_site, profile, run = onboarded(directory)
            with self.assertRaisesRegex(SchedulerConfigurationError, "run onboard-host"):
                onboard.ingest_phone(the_site, SiteProfile("site-a"), "phone-b", "model-a", [run],
                                     capacity=PhoneCapacity(1 << 40))
            two = RunDirectory(sweep_run(Path(directory) / "run-two", shares=(1600, 3200)))
            with self.assertRaisesRegex(LayerPlacementError, "at least three shares"):
                onboard.ingest_phone(the_site, profile, "phone-b", "model-a", [two], capacity=PhoneCapacity(1 << 40))
            with self.assertRaisesRegex(SchedulerConfigurationError, "no pair"):
                profile.pair("phone-b", "model-a")
            self.assertEqual(onboard.sweep_columns(9600, 3200), [3200, 6400, 9600])
            self.assertEqual(onboard.sweep_columns(12800, 400), [1600, 3200, 4800, 6400, 8000, 9600, 11200, 12800])
            with self.assertRaisesRegex(PhysicalAdapterError, "at least three shares"):
                onboard.sweep_columns(6400, 3200)


class SiteProfileTests(unittest.TestCase):
    def test_load_save_and_site_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "work" / "PROFILE.json"
            fresh = SiteProfile.load(path, "site-a")
            self.assertEqual((fresh.site_id, fresh.revision, fresh.pairs), ("site-a", 0, []))
            with self.assertRaisesRegex(SchedulerConfigurationError, "lacks"):
                fresh.set_host_model("model-a", {"n_ff": 1})
            with self.assertRaisesRegex(SchedulerConfigurationError, "no host record"):
                fresh.update_host_model("model-a", t_host_ms=1.0)
            fresh.save(path)
            self.assertEqual(SiteProfile.load(path, "site-a").to_json(), fresh.to_json())
            with self.assertRaisesRegex(SchedulerConfigurationError, "profile of site site-a, not site-b"):
                SiteProfile.load(path, "site-b")
            path.write_text("{")
            with self.assertRaisesRegex(SchedulerConfigurationError, "cannot be loaded"):
                SiteProfile.load(path, "site-a")
            with self.assertRaisesRegex(SchedulerConfigurationError, "not a site profile"):
                SiteProfile.from_json({"schema": "x"})


class FakeTcpSession:
    def __init__(self, configuration, **options):
        self.configuration, self.options, self.active, self.stops = configuration, options, False, []

    def preflight(self):
        return SimpleNamespace(to_json=lambda: {"phase": "preflight"})

    def start(self, log_path):
        self.active = True
        return SimpleNamespace(to_json=lambda: {"phase": "launch"})

    def transport_contract(self):
        return SimpleNamespace(server_environment=lambda: {"S41_SERVER_FFN_HOST": "127.0.0.1",
                                                           "S41_SERVER_FFN_PORT": "27321",
                                                           "S41_SERVER_FFN_TRANSPORT": "tcp"})

    def stop(self, **options):
        self.stops.append(options)
        self.active = False
        return SimpleNamespace(to_json=lambda: {"phase": "stop", **options})


def one_shard_binding(columns=3200) -> HelperBinding:
    bundle = "/data/local/tmp/bundle-b"
    return HelperBinding(ARTIFACT, 64, 6400, "swiglu",
                         (HelperShard("P0", "/data/local/tmp/site-a/model-a/P0.ffn.gguf", "sha256:" + "4" * 64,
                                      (0, 1, 2, 3), columns, 1000),),
                         {bundle + "/llama-ffn-split-worker": "sha256:" + "9" * 64})


class AdbTcpDriverTests(unittest.TestCase):
    def driver(self, *, listening=True, run=None):
        the_site = site()
        adb = AdbGuard(the_site.paths.adb_path, 5037, run=run or Recorder(), listening=lambda port: listening)
        return AdbTcpPhoneDriver(the_site.phone("phone-b"), the_site.paths, adb, session_factory=FakeTcpSession), the_site

    def test_an_unrooted_phone_runs_one_resident_worker_without_su(self):
        driver, the_site = self.driver()
        worker = driver.worker_configuration(one_shard_binding())
        self.assertEqual((worker.as_root, worker.max_requests, worker.backend, worker.phone_port, worker.forward_port),
                         (False, 0, "HTP0", 27321, 27321))
        self.assertEqual(worker.worker_environment["ADSP_LIBRARY_PATH"], "/data/local/tmp/bundle-b")
        command = driver.launch_command(one_shard_binding())
        self.assertFalse(command.startswith("su "))
        self.assertIn("LD_LIBRARY_PATH=/data/local/tmp/bundle-b", command)
        self.assertIn("--layers 0-3 --columns 3200", command)
        self.assertIsInstance(phone_driver(the_site, "phone-b"), AdbTcpPhoneDriver)
        self.assertIsNone(driver.power_calibration())                     # this phone's power is assumed

    def test_start_stop_and_release(self):
        with tempfile.TemporaryDirectory() as directory:
            driver, _ = self.driver()
            endpoint = driver.start(one_shard_binding(), Path(directory))
            self.assertEqual(endpoint.server_environment, {
                "S41_SERVER_FFN_HOST": "127.0.0.1", "S41_SERVER_FFN_PORT": "27321",
                "S41_SERVER_FFN_TRANSPORT": "tcp", "S41_SERVER_FFN_LAYER_MASK": "15"})
            self.assertEqual(endpoint.transport, "adb-tcp")
            stopped = driver.stop(served_calls=40)
            self.assertEqual((stopped["allow_idle_signal"], stopped["served_calls"]), (True, 40))
            self.assertEqual(driver.release(), [])
            driver.start(one_shard_binding(), Path(directory))
            self.assertEqual(driver.release(), [])
            self.assertEqual(driver.session.stops[-1], {"allow_idle_signal": True, "release_exited": True})

    def test_a_worker_started_without_root_is_found_by_its_exe_link(self):
        from research_dev.scheduler.adapters.phone_tcp_session import AdbTcpPhoneWorkerSession
        driver, _ = self.driver()
        worker = driver.worker_configuration(one_shard_binding())
        # Android's ps prints the command line without the path of its first word
        listing = ("  PID ARGS\n 4101 llama-ffn-split-worker -m /data/x.gguf\n"
                   " 4102 llama-ffn-split-worker -m /data/other.gguf\n 4103 sh -c other\n")
        run = Recorder({"ps -A -o PID,ARGS": listing, "readlink /proc/4101/exe": worker.worker_path + "\n",
                        "readlink /proc/4102/exe": "/data/local/tmp/someone-else/llama-ffn-split-worker\n"})
        session = AdbTcpPhoneWorkerSession(worker, run=run)
        self.assertEqual(session._worker_pids(), [4101])               # ours; the other path is someone else's
        self.assertFalse(any("su -c" in command[-1] for command in run.commands))   # no root on this phone
        self.assertEqual(session._signal_command(4101), "kill -TERM 4101")

    def test_refusals_and_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            driver, the_site = self.driver(listening=False)
            with self.assertRaises(AdbUnavailable):
                driver.start(one_shard_binding(), Path(directory))
            session = driver.build_session(one_shard_binding())
            with self.assertRaises(AdbUnavailable):
                session.options["run"]([str(the_site.paths.adb_path), "-P", "5037", "shell", "id"])
            with self.assertRaises(AdbUnavailable):
                session.options["popen"]([str(the_site.paths.adb_path), "shell", "-T", "worker"])
            two = HelperBinding(ARTIFACT, 64, 6400, "swiglu", (
                HelperShard("P0", "/data/local/tmp/site-a/a.gguf", "sha256:" + "4" * 64, (0,), 3200, 10),
                HelperShard("P1", "/data/local/tmp/site-a/b.gguf", "sha256:" + "5" * 64, (1,), 3200, 10)),
                one_shard_binding().pins)
            with self.assertRaisesRegex(PhysicalAdapterError, "one session"):
                driver.worker_configuration(two)
            with self.assertRaisesRegex(PhysicalAdapterError, "column quantum"):
                driver.worker_configuration(one_shard_binding(columns=1600))
            with self.assertRaisesRegex(PhysicalAdapterError, "not on the adb-tcp transport"):
                AdbTcpPhoneDriver(the_site.phone("phone-a"), the_site.paths, driver.adb)


if __name__ == "__main__":
    unittest.main()
