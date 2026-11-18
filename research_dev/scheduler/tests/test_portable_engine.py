"""Portable path: schedules, provisioning records and the run engine, with fake drivers (no hardware)."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.shared_layer import PhoneCapacity, SharedLayerFit, plan_static_share  # noqa: E402
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.campaigns.portable import engine, provision, schedule  # noqa: E402
from research_dev.scheduler.campaigns.portable.rundir import RunDirectory  # noqa: E402
from research_dev.scheduler.drivers.adb import AdbGuard  # noqa: E402
from research_dev.scheduler.drivers.host import HostLaunchPolicy, ServerHandle, ServerLoadFailure  # noqa: E402
from research_dev.scheduler.drivers.phone import HelperEndpoint  # noqa: E402

from test_drivers_base import Recorder  # noqa: E402
from test_drivers_runtime import ARTIFACT, BUNDLE, FakeProcess, binding, site  # noqa: E402

PROBE = [11, 12, 13]


class ScheduleTests(unittest.TestCase):
    def test_sweep_brackets_every_two_legs(self):
        plan = schedule.sweep("sweep-a", PROBE, [1600, 3200, 4800], output_tokens=16)
        self.assertEqual([phase["name"] for phase in plan["phases"]],
                         ["warm", "leg00c0", "leg01c1600", "leg02c3200", "leg03c0", "leg04c4800", "leg05c0"])
        self.assertEqual({request["id"] for phase in plan["phases"][1:] for request in phase["requests"]}, {"probe"})
        schedule.validate(plan, stored_columns=4800, column_quantum=1600)
        with self.assertRaisesRegex(schedule.ScheduleError, "positive column counts"):
            schedule.sweep("x", PROBE, [0])

    def test_seq_alternates_the_order_and_trace_scales_arrivals(self):
        requests = [{"id": f"request-{index}", "arrival_s": 100.0 + 10 * index, "prompt": [1, 2, index + 3],
                     "output_tokens": 8} for index in range(3)]
        seq = schedule.seq("seq-a", requests, 3200, PROBE, warm_tokens=4)
        self.assertEqual([phase["name"] for phase in seq["phases"]],
                         ["warm0", "warm3200", "r00c0", "r00c3200", "r01c3200", "r01c0", "r02c0", "r02c3200"])
        self.assertEqual(seq["phases"][4]["requests"][0]["id"], "request-1")
        trace = schedule.trace("trace-a", requests, 3200, PROBE, scale=0.5)
        self.assertEqual([request["arrival_s"] for request in trace["phases"][1]["requests"]], [0.0, 5.0, 10.0])
        self.assertEqual(trace["phases"][0]["requests"][0]["columns"], 3200)
        alone = schedule.trace("trace-b", requests, 0, PROBE)
        schedule.validate(alone, stored_columns=0, column_quantum=1)

    def test_rows_baseline_text_prompts_and_request_files(self):
        rows = schedule.rows("rows-a", "a probe text", [0, 3200], kmax=2, output_tokens=8)
        self.assertEqual([phase["name"] for phase in rows["phases"]], ["warm", "k1c0", "k1c3200", "k2c0", "k2c3200"])
        self.assertEqual(rows["phases"][4]["requests"][1], {"id": 1, "arrival_s": 0.0, "prompt_text": "a probe text",
                                                           "output_tokens": 8, "columns": 3200})
        baseline = schedule.host_baseline("host-a", schedule.PROBE_TEXT, legs=2)
        self.assertEqual([phase["name"] for phase in baseline["phases"]], ["warm", "leg00c0", "leg01c0"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in (
                {"event_id": "b", "arrival_us": 2_000_000, "prompt_tokens": [1], "output_tokens": 4},
                {"event_id": "a", "arrival_us": 1_000_000, "prompt_tokens": [2], "output_tokens": 5})) + "\n")
            loaded = schedule.load_requests(path)
        self.assertEqual([(row["id"], row["arrival_s"]) for row in loaded], [("a", 1.0), ("b", 2.0)])

    def test_validation_refuses_what_the_helper_cannot_serve(self):
        good = schedule.sweep("s", PROBE, [3200])
        for edit, message in (
                (lambda s: s["phases"][2]["requests"][0].update(columns=4800), "up to the stored 3200"),
                (lambda s: s["phases"][2]["requests"][0].update(columns=100), "multiples of 1600"),
                (lambda s: s["phases"][2]["requests"][0].update(output_tokens=1), "two output tokens"),
                (lambda s: s["phases"][2]["requests"][0].pop("prompt_tokens"), "prompt tokens or a prompt text"),
                (lambda s: s["phases"][2].update(name="warm"), "not unique"),
                (lambda s: s["phases"][2].update(requests=[]), "every phase needs a request")):
            plan = json.loads(json.dumps(good))
            edit(plan)
            with self.subTest(message=message), self.assertRaisesRegex(schedule.ScheduleError, message):
                schedule.validate(plan, stored_columns=3200, column_quantum=1600)
        with self.assertRaisesRegex(schedule.ScheduleError, "host alone"):
            schedule.validate(good, stored_columns=0, column_quantum=1)


def shard_index(columns=3200):
    return {"schema": "s42-ffn-shard-index-v1", "architecture": "qwen3", "n_embd": 64, "n_ff": 6400,
            "parent_sha256": ARTIFACT, "weight_type": "F16", "shards": [
                {"session_id": "S0", "path": "S0.ffn.gguf", "shard_sha256": "sha256:" + "4" * 64,
                 "layer_mask": "0000000000000003", "columns": columns, "shard_bytes": 1000},
                {"session_id": "S1", "path": "S1.ffn.gguf", "shard_sha256": "sha256:" + "5" * 64,
                 "layer_mask": "000000000000000c", "columns": columns, "shard_bytes": 1000}]}


class ProvisionTests(unittest.TestCase):
    def test_adopt_checks_the_phones_shards_and_pins_its_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "FFN_SHARDS.json"
            index.write_text(json.dumps(shard_index()))
            remote = "/data/local/tmp/site-a/shards"
            answers = {"sha256sum " + remote: f"{'4' * 64}  {remote}/S0.ffn.gguf\n{'5' * 64}  {remote}/S1.ffn.gguf\n",
                       "ls ": "llama-ffn-split-worker\nlibggml.so\n",
                       "sha256sum " + BUNDLE: f"{'9' * 64}  {BUNDLE}/llama-ffn-split-worker\n{'8' * 64}  {BUNDLE}/libggml.so\n"}
            run = Recorder(answers)
            adb = AdbGuard(Path("/opt/adb"), 5037, run=run, listening=lambda port: True)
            document = provision.adopt(site(), "phone-a", "model-a", index, remote, adb=adb)
            self.assertEqual((document["device_id"], document["model_id"], document["weight_type"]),
                             ("phone-a", "model-a", "F16"))
            path = Path(directory) / "bindings" / "binding.json"
            provision.save_binding(path, document)
            loaded, again = provision.load_binding(path)
            self.assertEqual((loaded.layers, loaded.stored_columns, loaded.activation), ((0, 1, 2, 3), 3200, "swiglu"))
            self.assertEqual(loaded.pins[BUNDLE + "/libggml.so"], "sha256:" + "8" * 64)
            self.assertEqual(again, document)
            self.assertFalse(any("push" in command or "rm " in command[-1] for command in run.commands))   # read-only
            changed = Recorder({**answers, "sha256sum " + remote: f"{'0' * 64}  {remote}/S0.ffn.gguf\n"})
            with self.assertRaisesRegex(PhysicalAdapterError, "differ from the index"):
                provision.adopt(site(), "phone-a", "model-a", index, remote,
                                adb=AdbGuard(Path("/opt/adb"), 5037, run=changed, listening=lambda port: True))
            with self.assertRaisesRegex(PhysicalAdapterError, "outside this site's directories"):
                provision.adopt(site(), "phone-a", "model-a", index, "/data/local/tmp/someone-else", adb=adb)
            (Path(directory) / "other.json").write_text(json.dumps({"schema": "x"}))
            with self.assertRaisesRegex(PhysicalAdapterError, "not a helper binding"):
                provision.load_binding(Path(directory) / "other.json")

    def test_shard_specs_follow_the_share_plan_and_slicing_respects_the_run_lock(self):
        fit = SharedLayerFit(1, 6400, 9.5, 0.3, 1.6, 13.0)
        plan = plan_static_share(fit, cpu_layers=range(4), n_embd=64, column_quantum=400, weight_type="F16",
                                 capacity=PhoneCapacity(10_000_000, sessions=2, session_limit_bytes=2_000_000))
        self.assertEqual(provision.shard_specs(plan), ["S0=0-1:2400", "S1=2-3:2400"])
        none = plan_static_share(SharedLayerFit(1, 6400, 5.0, 0.3, 6.0, 13.0), cpu_layers=range(4), n_embd=64,
                                 column_quantum=400, weight_type="F16", capacity=PhoneCapacity(10_000_000))
        with self.assertRaisesRegex(PhysicalAdapterError, "keeps nothing"):
            provision.shard_specs(none)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "shards"
            calls = []

            def run(command, **keywords):
                calls.append(command)
                (out / "FFN_SHARDS.json").write_text(json.dumps(shard_index(2400)))

            the_site = site(directory)
            index = provision.slice_shards(the_site, "model-a", ARTIFACT, ["S0=0-1:2400"], out, run=run,
                                           lock_held=lambda root: False)
            self.assertEqual(index["shards"][0]["columns"], 2400)
            self.assertEqual(calls[0][-2:], ["--shard", "S0=0-1:2400"])
            self.assertIn("native/ffn_shard_gguf.py", calls[0][1])
            with self.assertRaisesRegex(PhysicalAdapterError, "metered run holds the site lock"):
                provision.slice_shards(the_site, "model-a", ARTIFACT, ["S0=0-1:2400"], out, run=run,
                                       lock_held=lambda root: True)
            self.assertEqual(len(calls), 1)


class Clock:
    """A fake monotonic clock that ``sleep`` advances (threads share it)."""

    def __init__(self):
        self.now, self._lock = 1_000_000_000_000, threading.Lock()

    def ns(self):
        with self._lock:
            self.now += 1_000            # every reading moves on, so windows are never empty
            return self.now

    def sleep(self, seconds):
        with self._lock:
            self.now += int(seconds * 1e9)
        time.sleep(0)                    # let the request threads run


class FakeWatch:
    def __init__(self, disturbances=()):
        self.samples, self.tripped, self.rearmed, self.stopped = [], None, 0, 0
        self.disturbances = list(disturbances)

    def start(self):
        pass

    def stop(self):
        self.stopped += 1

    def rearm(self):
        self.rearmed += 1

    def quiet(self, below, seconds):
        return True

    def recent(self, seconds):
        return [0.0]

    def summary(self, start, end):
        return {"samples": 1, "mean_cores": 0.0}

    def disturbance(self, start, end, quiet_cores):
        return self.disturbances.pop(0) if self.disturbances else {"samples": 3, "max_cores": 0.5, "mean_excess_cores": 0.0}


class FakeMeter:
    def __init__(self, events):
        self.events = events

    def start(self):
        self.events.append("meter.start")

    def stop(self):
        self.events.append("meter.stop")

    def wait_past(self, instant, timeout=15.0):
        pass

    def energy(self, start, end):
        return {"cpu_package_energy_j": 1.0, "gpu_board_energy_j": 1.0, "cpu_package_average_power_w": 100.0,
                "gpu_board_average_power_w": 50.0}

    def save(self, directory):
        self.events.append("meter.save")
        (Path(directory) / "POWER_SAMPLES.json").write_text('[{"t_ns": 1}]')


class FakeHost:
    def __init__(self, events, *, watch=None, exit_status=0, phone_calls=12, ready=None):
        self.events, self.watch, self.exit_status, self.phone_calls = events, watch or FakeWatch(), exit_status, phone_calls
        self.ready, self.erased, self.launched, self.killed, self.trip = ready, [], [], 0, None

    def meter(self, *, interval_s=0.1):
        return FakeMeter(self.events)

    def foreign_load(self, own_pids, *, limit_cores, window_s, on_trip):
        self.trip = on_trip
        return self.watch

    def launch(self, model_id, alias, port, policy, *, binding, helper_environment, output):
        self.events.append("server.launch")
        self.launched.append((model_id, binding, dict(helper_environment)))
        return ServerHandle(FakeProcess(), port, ("server", "-m", model_id), dict(helper_environment),
                            Path(output) / "server.log", policy.parallel)

    def wait_ready(self, handle, timeout_s):
        if self.ready is not None:
            raise self.ready

    def tokenize(self, handle, text):
        self.events.append("tokenize")
        return [7, 8, 9]

    def slots_busy(self, handle):
        return []

    def erase_slot(self, handle, slot, timeout_s=20):
        self.erased.append(slot)
        return {"id_slot": slot, "n_erased": 5}

    def stop(self, handle):
        self.events.append("server.stop")
        handle.process.returncode = self.exit_status
        return {"exit_status": self.exit_status, "phone_calls": self.phone_calls,
                "shape_lines": ['S41SERVERFFNSHAPE {"tokens":1}']}

    def kill(self, handle):
        self.killed += 1
        handle.process.returncode = -9


class FakePhone:
    def __init__(self, events, *, stop_error=None):
        self.events, self.stop_error, self.receipts, self.released, self.served = events, stop_error, {}, 0, None

    def start(self, binding_, log_directory):
        self.events.append("phone.start")
        self.receipts = {"preflight": {"phase": "preflight"}, "launch": {"phase": "launch"}, "launch_dropped": None}
        return HelperEndpoint({"S41_SERVER_FFN_PORT": "27391"}, binding_.layers, binding_.stored_columns,
                              "functionfs-bridge")

    def stop(self, *, served_calls):
        self.events.append("phone.stop")
        self.served = served_calls
        if self.stop_error is not None:
            raise self.stop_error
        return {"phase": "stop", "served_calls": served_calls}

    def release(self):
        self.events.append("phone.release")
        self.released += 1
        return []


class FakePhoneMeter:
    def __init__(self, events):
        self.events = events

    def start(self, run_directory, token):
        self.events.append("phone-meter.start")
        return {}

    def collect(self, run_directory):
        self.events.append("phone-meter.collect")
        return 0


class FakeClient:
    controls: list = []
    fail_on: str | None = None

    def complete(self, endpoint, payload, check):
        if FakeClient.fail_on and FakeClient.fail_on in payload.request_id:
            raise RuntimeError("connection reset")
        check()
        instant = time.monotonic_ns()
        payload.on_first_token(instant)
        for predicted in range(1, payload.output_tokens):
            payload.on_decode_progress(3, predicted, instant + predicted * 1000, predicted == payload.output_tokens - 1)
        return {"tokens": list(range(payload.output_tokens)), "prompt_ms": 1.0, "predicted_ms": 2.0,
                "endpoint_slot_probe_errors": []}

    def apply_ffn_control(self, endpoint, control, *, timeout_s=5):
        FakeClient.controls.append(control)
        return {"applied_token_index": 1, "slot_id": 3}, time.monotonic_ns()


def run_plan(shared=True):
    the_binding = binding() if shared else None
    return engine.RunPlan("model-a", "alias-a", 27431, HostLaunchPolicy(47, 8192, 4), 64, 6400, ARTIFACT,
                          "phone-a" if shared else None, the_binding, idle_s=1.0, tail_idle_s=1.0)


class EngineTests(unittest.TestCase):
    def setUp(self):
        FakeClient.controls, FakeClient.fail_on = [], None

    def run_engine(self, directory, plan, the_schedule, *, host=None, phone=None, options=None):
        events: list = []
        clock = Clock()
        host = host or FakeHost(events)
        host.events = events
        if phone is not None:
            phone.events = events
        phone = phone if phone is not None or not plan.shared else FakePhone(events)
        options = options or engine.RunOptions()
        options.log = lambda *words: events.append("log:" + " ".join(str(word) for word in words))
        output = Path(directory) / "run"
        the_engine = engine.Engine(site(directory), plan, the_schedule, output, options, host=host, phone=phone,
                                   host_meter=FakeMeter(events), helper_meter=FakePhoneMeter(events) if plan.shared else None,
                                   client_factory=FakeClient, clock_ns=clock.ns, sleep=clock.sleep)
        return the_engine, output, events, host, phone

    def test_alone_run_never_touches_a_phone(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = run_plan(shared=True).alone()
            the_schedule = schedule.host_baseline("host-a", "a probe text", legs=2, output_tokens=6, gap_s=0)
            the_engine, output, events, host, phone = self.run_engine(directory, plan, the_schedule)
            result = the_engine.run()
            self.assertEqual((result["status"], result["mode"]), ("PASS", "alone"))
            self.assertIsNone(phone)
            self.assertFalse(any(event.startswith("phone") for event in events))
            self.assertEqual(host.launched[0][1:], (None, {}))
            self.assertEqual(events.count("tokenize"), 1)                  # one text, tokenised once
            saved = json.loads((output / "SCHEDULE.json").read_text())
            self.assertEqual(saved["phases"][1]["requests"][0]["prompt_tokens"], [7, 8, 9])
            self.assertEqual(sorted(path.name for path in output.glob("PHASE-*.json")),
                             ["PHASE-0.json", "PHASE-1.json", "PHASE-2.json"])
            self.assertEqual(host.erased, [3, 3, 3])                       # every finished request's slot
            self.assertEqual(FakeClient.controls, [])
            for name in ("CONFIG.json", "IDENTITY.json", "IDLE.json", "TAIL_IDLE.json", "FOREIGN_LOAD.json",
                         "RESULT.json", "SERVER_FFN_SHAPES.json", "POWER_SAMPLES.json"):
                self.assertTrue((output / name).is_file(), name)
            config = json.loads((output / "CONFIG.json").read_text())
            self.assertEqual((config["helper"], config["phone"]["columns"], config["site_id"]), (None, 6400, "site-a"))
            phase = json.loads((output / "PHASE-1.json").read_text())
            self.assertEqual((phase["attempts"], phase["requests"][0]["request_id"]), (1, "host-a-leg00c0-00"))
            self.assertEqual(events.index("meter.stop") < events.index("meter.save"), True)
            self.assertFalse((output.parent / "portable-run.lock").exists() and
                             __import__("research_dev.scheduler.drivers.doctor", fromlist=["RunLock"]).RunLock.held(output.parent))

    def test_shared_run_applies_the_share_at_the_first_token_and_closes_the_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            the_schedule = schedule.sweep("sweep-a", PROBE, [3200, 3200], output_tokens=6, gap_s=0)
            the_engine, output, events, host, phone = self.run_engine(directory, run_plan(), the_schedule)
            result = the_engine.run()
            self.assertEqual((result["status"], result["mode"], result["phone_calls"]), ("PASS", "shared", 12))
            order = [event for event in events if not event.startswith("log:")]
            self.assertEqual(order[:3], ["phone-meter.start", "phone.start", "server.launch"])   # sampler, helper, server
            self.assertEqual(order[-5:], ["server.stop", "phone.stop", "meter.stop", "meter.save", "phone-meter.collect"])
            self.assertEqual(phone.served, 12)
            self.assertEqual([control.policy.columns for control in FakeClient.controls], [3200, 3200])
            control = FakeClient.controls[1]
            self.assertEqual((control.policy.layer_indices, control.policy.split_fraction_ppm, control.slot_id),
                             ((0, 1, 2, 3), 500000, 3))
            self.assertEqual(host.launched[0][2], {"S41_SERVER_FFN_PORT": "27391"})
            self.assertEqual(json.loads((output / "HELPER_LAUNCH.json").read_text()), {"phase": "launch"})
            self.assertEqual(json.loads((output / "HELPER_STOP.json").read_text())["served_calls"], 12)
            config = json.loads((output / "CONFIG.json").read_text())
            self.assertEqual((config["helper"]["kind"], config["helper"]["layers"], config["helper"]["columns"]),
                             ("functionfs-bridge", [0, 1, 2, 3], 3200))
            shared_phase = json.loads((output / "PHASE-2.json").read_text())["requests"][0]
            self.assertEqual((shared_phase["columns"], shared_phase["control"]["applied_token_index"]), (3200, 1))
            loaded = RunDirectory(output)                                 # the loader accepts the engine's files
            self.assertEqual((loaded.shared, loaded.n_ff, loaded.layers_on_phone, len(loaded.phases)), (True, 6400, 4, 5))
            self.assertEqual([row["columns"] for row in loaded.metered_phases()[1]["requests"]], [3200])

    def test_disturbed_phase_is_recorded_and_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            watch = FakeWatch(disturbances=[{"samples": 3, "max_cores": 0.0, "mean_excess_cores": 0.0},     # warm
                                            {"samples": 3, "max_cores": 90.0, "mean_excess_cores": 30.0}])  # leg 0
            the_schedule = schedule.host_baseline("host-a", PROBE, legs=1, output_tokens=6, gap_s=0)
            the_engine, output, events, host, _ = self.run_engine(directory, run_plan(False), the_schedule,
                                                                  host=FakeHost([], watch=watch))
            the_engine.run()
            disturbed = json.loads((output / "PHASE-1.disturbed-0.json").read_text())
            self.assertEqual((disturbed["cancelled"], disturbed["foreign"]["mean_excess_cores"]), (False, 30.0))
            phase = json.loads((output / "PHASE-1.json").read_text())
            self.assertEqual((phase["attempts"], phase["requests"][0]["request_id"]), (2, "host-a-leg00c0-00-r1"))
            self.assertTrue(any(event.startswith("log:PHASE_DISTURBED leg00c0 attempt 0 finished") for event in events))
            self.assertEqual(len(RunDirectory(output).phases), 2)          # the disturbed attempt is not a phase

    def test_too_many_disturbances_end_the_run_as_host_busy(self):
        with tempfile.TemporaryDirectory() as directory:
            noisy = [{"samples": 3, "max_cores": 90.0, "mean_excess_cores": 30.0}] * 5
            the_schedule = schedule.host_baseline("host-a", PROBE, legs=1, output_tokens=6, gap_s=0)
            the_engine, output, events, host, _ = self.run_engine(
                directory, run_plan(False), the_schedule, host=FakeHost([], watch=FakeWatch(disturbances=noisy)),
                options=engine.RunOptions(max_attempts=2))
            with self.assertRaisesRegex(engine.HostBusy, "disturbed 2 times"):
                the_engine.run()
            failure = json.loads((output / "FAILURE.json").read_text())
            self.assertEqual((failure["kind"], failure["helper_cleanup"]), ("HostBusy", ["server killed"]))
            self.assertFalse((output / "RESULT.json").exists())

    def test_failed_request_ends_the_server_and_releases_the_phone(self):
        with tempfile.TemporaryDirectory() as directory:
            FakeClient.fail_on = "leg01c3200"
            the_schedule = schedule.sweep("sweep-a", PROBE, [3200], output_tokens=6, gap_s=0)
            the_engine, output, events, host, phone = self.run_engine(directory, run_plan(), the_schedule)
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                the_engine.run()
            self.assertEqual((host.killed, phone.released), (1, 1))
            failure = json.loads((output / "FAILURE.json").read_text())
            self.assertIn("connection reset", failure["error"])
            self.assertEqual(failure["helper_cleanup"], ["server killed", "phone released"])
            order = [event for event in events if not event.startswith("log:")]
            self.assertEqual(order[-4:], ["phone.release", "meter.stop", "meter.save", "phone-meter.collect"])

    def test_load_failure_keeps_its_kind_and_releases_the_phone(self):
        with tempfile.TemporaryDirectory() as directory:
            host = FakeHost([], ready=ServerLoadFailure("gpu-out-of-memory", "exit status 1"))
            the_schedule = schedule.sweep("sweep-a", PROBE, [3200], output_tokens=6, gap_s=0)
            the_engine, output, events, host, phone = self.run_engine(directory, run_plan(), the_schedule, host=host)
            with self.assertRaises(ServerLoadFailure):
                the_engine.run()
            self.assertEqual(json.loads((output / "FAILURE.json").read_text())["kind"], "gpu-out-of-memory")
            self.assertEqual(phone.released, 1)

    def test_helper_stop_error_keeps_the_result(self):
        with tempfile.TemporaryDirectory() as directory:
            phone = FakePhone([], stop_error=RuntimeError("phone session exit status 1"))
            the_schedule = schedule.sweep("sweep-a", PROBE, [3200], output_tokens=6, gap_s=0)
            the_engine, output, events, host, phone = self.run_engine(directory, run_plan(), the_schedule, phone=phone)
            self.assertEqual(the_engine.run()["status"], "PASS")
            error = json.loads((output / "HELPER_STOP_ERROR.json").read_text())
            self.assertIn("exit status 1", error["error"])
            self.assertEqual(phone.released, 1)
            self.assertTrue((output / "RESULT.json").is_file())

    def test_schedule_that_the_helper_cannot_serve_starts_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            the_schedule = schedule.sweep("sweep-a", PROBE, [6400], output_tokens=6)
            the_engine, output, events, host, phone = self.run_engine(directory, run_plan(), the_schedule)
            with self.assertRaisesRegex(schedule.ScheduleError, "stored 3200"):
                the_engine.run()
            self.assertEqual((events, output.exists()), ([], False))


class RunPlanTests(unittest.TestCase):
    def test_round_trip_alone_view_and_refusals(self):
        plan = run_plan()
        self.assertEqual(engine.RunPlan.from_json(json.loads(json.dumps(plan.to_json()))), plan)
        alone = plan.alone()
        self.assertEqual((alone.shared, alone.device_id, alone.n_ff, alone.policy), (False, None, 6400, plan.policy))
        with self.assertRaisesRegex(PhysicalAdapterError, "both the helper phone and its binding"):
            engine.RunPlan("m", "a", 1, plan.policy, 64, 6400, ARTIFACT, "phone-a", None)
        with self.assertRaisesRegex(PhysicalAdapterError, "another model"):
            engine.RunPlan("m", "a", 1, plan.policy, 64, 12800, ARTIFACT, "phone-a", binding())
        with self.assertRaisesRegex(PhysicalAdapterError, "slot-save path"):
            engine.RunPlan("m", "a", 1, HostLaunchPolicy(1, 8, 1, slot_save=False), 64, 6400, ARTIFACT)

    def test_dry_run_starts_nothing(self):
        events: list = []

        class Host(FakeHost):
            def command(self, model_id, alias, port, policy, *, binding, helper_environment, output):
                return ["server", model_id], dict(helper_environment)

        phone = SimpleNamespace(launch_command=lambda binding_: "phone launch")
        result = engine.dry_run(site(), run_plan(), Path("/work/run"), host=Host(events), phone=phone)
        self.assertEqual(result["phone_launch"], "phone launch")
        self.assertEqual(result["environment"]["S41_SERVER_FFN_PORT"], "27391")
        self.assertEqual(result["environment"]["S41_SERVER_FFN_LAYER_MASK"], "15")
        self.assertEqual(events, [])
        alone = engine.dry_run(site(), run_plan().alone(), Path("/work/run"), host=Host(events))
        self.assertEqual((alone["environment"], alone["phone_launch"]), ({}, None))


if __name__ == "__main__":
    unittest.main()
