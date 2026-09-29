#!/usr/bin/env python3
"""Opt-in scheduler-driven device power control (campaign ``device_power``).

The controller lowers the desktop GPU SM clock (and optionally the CPU EPP) while the scheduler
knows the GPU has no work, restores before the next known work, caps the clock during decode and
locks it at the floor during a model load. Every command is the exact ``sudo -n`` argv the desktop
sudoers rule permits; the first failure flips the controller to UNAVAILABLE. Without the policy
nothing is constructed and every RESULT stays byte-identical.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler.adapters import PhysicalAdapterError
from research_dev.scheduler.adapters import host_runtime
from research_dev.scheduler.adapters.device_power import (
    CpuEppControl,
    DevicePowerController,
    DevicePowerPolicy,
    NvidiaClockControl,
    device_power_capability,
)
from research_dev.scheduler.adapters.heterogeneous_rig import rig_device_power_policy
from research_dev.scheduler.campaigns.burstgpt import arguments, preflight, runner
from research_dev.scheduler.campaigns.burstgpt.tools import device_power_energy
from research_dev.scheduler.config import (
    CAMPAIGN_MANIFEST_SCHEMA,
    CampaignManifest,
    DevicePowerConfiguration,
    SchedulerConfigurationError,
)

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from test_llama_server_adapter import execution_command  # noqa: E402

UUID = "GPU-0123abcd-0000-0000-0000-000000000001"
SMI = "/usr/bin/nvidia-smi"
TEE = "/usr/bin/tee"
IDLE = {"min_gap_s": 60, "lead_ms": 500, "gpu_min_clocks_mhz": 210, "cpu_epp": "power"}
POLICY = {"device": "desktop-cuda", "gpu_uuid": UUID, "idle": IDLE}
ARTIFACT = "sha256:" + "a" * 64
S = 1_000_000  # us per second


def policy(**changes) -> DevicePowerConfiguration:
    return DevicePowerConfiguration.from_json({**POLICY, **changes})


def gpu_command(ticket_id="req-1:attempt:0", executor_id="physical:desktop-gpu", device_id="desktop-cuda"):
    base = execution_command(ARTIFACT)
    participant = replace(base.participants[1], device_id=device_id, executor_id=executor_id)
    return replace(base, ticket_id=ticket_id, executor_id=executor_id, participants=(base.participants[0], participant))


class RecordedRun:
    """``subprocess.run`` stand-in: records every argv, answers ``sudo -l`` and the clock query,
    tracks the locked SM clock and writes ``tee`` input into the (temporary) EPP files."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.failures: dict[tuple[str, ...], object] = {}  # argv prefix -> returncode or exception
        self.sm_mhz = 2505

    def __call__(self, argv, *, capture_output, text, timeout, check, input=None):
        assert capture_output is True and text is True and timeout == 5.0 and check is False
        self.calls.append(list(argv))
        for prefix, outcome in self.failures.items():
            if tuple(argv[:len(prefix)]) == prefix:
                if isinstance(outcome, BaseException):
                    raise outcome
                return SimpleNamespace(returncode=outcome, stdout="", stderr="refused\n")
        if argv[:3] == ["sudo", "-n", "-l"]:
            return SimpleNamespace(returncode=0, stdout=" ".join(argv[3:]) + "\n", stderr="")
        if "-lgc" in argv:
            self.sm_mhz = int(argv[argv.index("-lgc") + 1].split(",")[0])
            return SimpleNamespace(returncode=0, stdout="GPU clocks set to ...\n", stderr="")
        if "-rgc" in argv:
            self.sm_mhz = 2505
            return SimpleNamespace(returncode=0, stdout="All done.\n", stderr="")
        if "--query-gpu=clocks.sm,clocks.mem,pstate" in argv:
            return SimpleNamespace(returncode=0, stdout=f"{self.sm_mhz}, 405, P8\n", stderr="")
        if TEE in argv:
            for path in argv[argv.index(TEE) + 1:]:
                Path(path).write_text(input)
            return SimpleNamespace(returncode=0, stdout=input, stderr="")
        raise AssertionError("unexpected command " + repr(argv))

    def since(self, index: int) -> list[list[str]]:
        return self.calls[index:]


class ControllerHarness(unittest.TestCase):
    """A controller with a recorded ``run``, a fake clock and a two-CPU EPP tree."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="device-power-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.cpu_root = self.root / "cpu"
        for name in ("cpu0", "cpu1"):
            (self.cpu_root / name / "cpufreq").mkdir(parents=True)
            (self.cpu_root / name / "cpufreq" / "energy_performance_preference").write_text("balance_performance\n")
        (self.cpu_root / "cpuidle").mkdir()  # not a CPU
        self.output = self.root / "out"
        self.output.mkdir()
        self.run = RecordedRun()
        self.now_ns = [10 ** 12]
        self.epoch_ns = 10 ** 12

    @property
    def epp_paths(self):
        return [str(self.cpu_root / name / "cpufreq" / "energy_performance_preference") for name in ("cpu0", "cpu1")]

    def epp_values(self):
        return [Path(path).read_text().strip() for path in self.epp_paths]

    def advance(self, seconds: float) -> None:
        self.now_ns[0] += int(seconds * 1e9)

    def controller(self, configuration=None, *, tick_interval_s=0.5, probe=True) -> DevicePowerController:
        configuration = policy() if configuration is None else configuration
        clock = NvidiaClockControl(configuration.gpu_uuid, run=self.run, monotonic_ns=lambda: self.now_ns[0])
        epp = (CpuEppControl(run=self.run, monotonic_ns=lambda: self.now_ns[0], cpu_root=self.cpu_root)
               if configuration.idle is not None and configuration.idle.cpu_epp is not None else None)
        controller = DevicePowerController(
            configuration, epoch_ns_provider=lambda: self.epoch_ns, output_directory=self.output,
            run=self.run, clock=clock, epp=epp, monotonic_ns=lambda: self.now_ns[0], tick_interval_s=tick_interval_s)
        if probe:
            self.assertTrue(controller.capability_probe())
            self.assertEqual(controller.state, "RESTORED")
        return controller

    def transitions(self, controller):
        return [(row["from"], row["to"], row["reason"]) for row in controller.events]


LGC_MIN = ["sudo", "-n", SMI, "-lgc", "210,210", "-i", UUID]
RGC = ["sudo", "-n", SMI, "-rgc"]
QUERY = [SMI, "-i", UUID, "--query-gpu=clocks.sm,clocks.mem,pstate", "--format=csv,noheader,nounits"]


class PolicyValidationTests(unittest.TestCase):
    def test_round_trip_and_omitted_optionals(self):
        configuration = policy()
        self.assertEqual(configuration.to_json(), {"device": "desktop-cuda", "gpu_uuid": UUID, "idle": {
            "cpu_epp": "power", "gpu_min_clocks_mhz": 210, "lead_ms": 500, "min_gap_s": 60}})
        self.assertEqual(DevicePowerConfiguration.from_json(configuration.to_json()), configuration)
        full = policy(decode_cap={"sm_max_mhz": 1200}, load_min=True)
        self.assertEqual(full.to_json()["decode_cap"], {"sm_max_mhz": 1200})
        self.assertIs(full.to_json()["load_min"], True)
        self.assertEqual(DevicePowerConfiguration.from_json(full.to_json()), full)
        self.assertIsNone(policy(idle={**IDLE, "cpu_epp": None}).idle.cpu_epp)
        self.assertFalse(policy(load_min=False).load_min)
        self.assertIs(DevicePowerPolicy, DevicePowerConfiguration)

    def test_rejects_untyped_and_unknown_values(self):
        for name, value in (("min_gap_s", True), ("min_gap_s", 60.0), ("min_gap_s", "60"), ("min_gap_s", 0),
                            ("lead_ms", False), ("lead_ms", 60_000), ("gpu_min_clocks_mhz", 210.0),
                            ("gpu_min_clocks_mhz", 0), ("cpu_epp", "eco"), ("cpu_epp", True), ("cpu_epp", 1)):
            with self.subTest(name=name, value=value), self.assertRaises(SchedulerConfigurationError):
                policy(idle={**IDLE, name: value})
        invalid = (
            {"idle": {**IDLE, "extra": 1}}, {"idle": {"min_gap_s": 60}}, {"idle": []}, {"unknown": 1},
            {"device": ""}, {"device": 7}, {"gpu_uuid": None}, {"load_min": 1}, {"load_min": "true"},
            {"decode_cap": {"sm_max_mhz": 1200, "x": 1}}, {"decode_cap": {"sm_max_mhz": True}},
            {"decode_cap": {"sm_max_mhz": 210}}, {"decode_cap": {"sm_max_mhz": 100}},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(SchedulerConfigurationError):
                policy(**changes)
        # no behaviour, and decode_cap / load_min without the idle floor
        for row in ({"device": "desktop-cuda", "gpu_uuid": UUID},
                    {"device": "desktop-cuda", "gpu_uuid": UUID, "decode_cap": {"sm_max_mhz": 1200}},
                    {"device": "desktop-cuda", "gpu_uuid": UUID, "load_min": True},
                    {"gpu_uuid": UUID, "idle": IDLE}, [POLICY], "desktop-cuda"):
            with self.subTest(row=row), self.assertRaises(SchedulerConfigurationError):
                DevicePowerConfiguration.from_json(row)


class ManifestAndArgumentTests(unittest.TestCase):
    ROW = {"schema": CAMPAIGN_MANIFEST_SCHEMA, "campaign_id": "device-power-test",
           "rig_manifest_path": "rig.json", "models_manifest_path": "models.json",
           "evidence_manifest_path": "evidence.json", "selection_mode": "energy-aware",
           "maximum_latency_ppm": 1_100_000, "energy_attribution_kind": "diagnostic",
           "trace": {"large_requests_path": "large.json", "overlay_requests_path": "overlay.json",
                     "trace_manifest_path": "trace.json"}}

    def test_campaign_manifest_omits_the_key_by_default_and_round_trips_when_set(self):
        plain = CampaignManifest.from_json(dict(self.ROW), Path("/inputs"))
        self.assertIsNone(plain.device_power)
        self.assertNotIn("device_power", plain.to_json())
        configured = CampaignManifest.from_json({**self.ROW, "device_power": POLICY}, Path("/inputs"))
        self.assertEqual(configured.device_power, policy())
        self.assertEqual(configured.to_json()["device_power"], policy().to_json())
        self.assertEqual(CampaignManifest.from_json(configured.to_json(), Path("/inputs")), configured)
        self.assertEqual(configured.to_json().keys() - plain.to_json().keys(), {"device_power"})
        for invalid in ({**POLICY, "idle": {**IDLE, "min_gap_s": 1.5}}, {"device": "desktop-cuda"}, [POLICY], True):
            with self.subTest(invalid=invalid), self.assertRaises(SchedulerConfigurationError):
                CampaignManifest.from_json({**self.ROW, "device_power": invalid}, Path("/inputs"))

    def test_runner_and_preflight_arguments_parse_the_same_object(self):
        parsed = arguments.device_power_json(json.dumps(POLICY))
        self.assertEqual(parsed, policy().to_json())
        self.assertIs(type(parsed), dict)
        for text in ("null", "[]", "{", '{"device": "desktop-cuda"}', json.dumps({**POLICY, "load_min": 1})):
            with self.subTest(text=text), self.assertRaises((argparse.ArgumentTypeError, ValueError)):
                arguments.device_power_json(text)
        for parser in (arguments._build_parser(),):
            action = next(row for row in parser._actions if "--device-power-json" in row.option_strings)
            self.assertEqual(action.dest, "device_power")
            self.assertIsNone(action.default)
        source = Path(preflight.__file__).read_text(encoding="utf-8")
        self.assertIn('"--device-power-json", dest="device_power", type=device_power_json', source)

    def test_launch_commands_carry_the_flag_only_when_declared(self):
        from test_two_phone_helpers import CampaignConfigurationTests, two_phone_rig_json
        from research_dev.scheduler.campaigns.burstgpt import launch
        from research_dev.scheduler.config import load_scheduler_configuration

        case = CampaignConfigurationTests()
        case.setUp()
        self.addCleanup(case.tearDown)
        path = case._write(two_phone_rig_json(), None)
        plain = load_scheduler_configuration(path, environ={})
        arguments_of = lambda configuration: launch.preflight_command(  # noqa: E731
            configuration, catalog_path=case.root / "c.json", normal_usb_receipt_path=case.root / "u.json",
            output_path=case.root / "out.json")
        self.assertNotIn("--device-power-json", arguments_of(plain))
        campaign = json.loads(path.read_text())
        campaign["device_power"] = POLICY
        path.write_text(json.dumps(campaign))
        configured = load_scheduler_configuration(path, environ={})
        command = arguments_of(configured)
        value = json.loads(command[command.index("--device-power-json") + 1])
        self.assertEqual(value, policy().to_json())
        source = Path(launch.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count('"--device-power-json"'), 2)  # runner_command and preflight_command


class ControlArgvTests(ControllerHarness):
    def test_nvidia_control_issues_the_exact_sudoers_argv_and_reads_back(self):
        control = NvidiaClockControl(UUID, run=self.run, monotonic_ns=lambda: self.now_ns[0])
        result = control.lock(210, 210)
        self.assertEqual(self.run.calls, [LGC_MIN, QUERY])
        self.assertEqual(result["command"], LGC_MIN)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["readback"], {"clocks_sm_mhz": 210, "clocks_mem_mhz": 405, "pstate": "P8"})
        self.assertEqual(set(result), {"command", "returncode", "stderr", "stdout", "readback", "duration_us"})
        self.assertEqual(control.lock(210, 1200)["command"], ["sudo", "-n", SMI, "-lgc", "210,1200", "-i", UUID])
        self.assertEqual(control.restore()["command"], RGC)
        self.assertEqual(control.restore()["readback"]["clocks_sm_mhz"], 2505)
        probe = control.probe([(210, 210), (210, 1200)])
        self.assertEqual([row["command"] for row in probe], [
            ["sudo", "-n", "-l", SMI, "-lgc", "210,210", "-i", UUID],
            ["sudo", "-n", "-l", SMI, "-lgc", "210,1200", "-i", UUID],
            ["sudo", "-n", "-l", SMI, "-rgc"],
        ])
        self.assertTrue(all(row["returncode"] == 0 for row in probe))
        self.assertFalse(any("-lmc" in call or "-rmc" in call for call in self.run.calls))
        for pair in ((0, 210), (300, 210), (210.0, 210), (True, 210)):
            with self.subTest(pair=pair), self.assertRaises(PhysicalAdapterError):
                control.lock(*pair)

    def test_nvidia_failures_are_rows_not_exceptions(self):
        control = NvidiaClockControl(UUID, run=self.run, monotonic_ns=lambda: self.now_ns[0])
        self.run.failures[("sudo", "-n", SMI, "-lgc")] = 1
        result = control.lock(210, 210)
        self.assertEqual((result["returncode"], result["stderr"], result["readback"]), (1, "refused", None))
        self.run.failures[("sudo", "-n", SMI, "-rgc")] = subprocess.TimeoutExpired(SMI, 5)
        result = control.restore()
        self.assertIsNone(result["returncode"])
        self.assertTrue(result["stderr"].startswith("TimeoutExpired"))
        self.run.failures[("sudo", "-n", SMI, "-rgc")] = FileNotFoundError("sudo")
        self.assertTrue(control.restore()["stderr"].startswith("FileNotFoundError"))
        self.run.failures[(SMI, "-i")] = 3
        self.assertIn("error", control.readback())

    def test_epp_control_targets_every_cpu_and_restores_the_read_value(self):
        control = CpuEppControl(run=self.run, monotonic_ns=lambda: self.now_ns[0], cpu_root=self.cpu_root)
        self.assertEqual(list(control.targets()), self.epp_paths)
        self.assertEqual(control.read(), "balance_performance")
        (probe,) = control.probe()
        self.assertEqual(probe["command"], ["sudo", "-n", "-l", TEE, *self.epp_paths])
        result = control.write("power")
        self.assertEqual(result["command"], ["sudo", "-n", TEE, *self.epp_paths])
        self.assertEqual(result["readback"], "power")
        self.assertEqual(self.epp_values(), ["power", "power"])
        with self.assertRaises(PhysicalAdapterError):
            control.write("eco")
        empty = CpuEppControl(run=self.run, cpu_root=self.root / "absent")
        self.assertEqual(empty.targets(), ())
        self.assertIsNone(empty.read())
        self.assertIsNone(empty.probe()[0]["returncode"])


class ControllerStateMachineTests(ControllerHarness):
    def test_capability_probe_checks_exact_commands_then_restores_to_a_known_state(self):
        controller = self.controller()
        self.assertEqual(self.run.calls, [
            ["sudo", "-n", "-l", SMI, "-lgc", "210,210", "-i", UUID],
            ["sudo", "-n", "-l", SMI, "-rgc"],
            ["sudo", "-n", "-l", TEE, *self.epp_paths],
            RGC, QUERY,
        ])
        (event,) = controller.events
        self.assertEqual({key: event[key] for key in ("kind", "device", "from", "to", "reason", "at_us")},
                         {"kind": "DEVICE_POWER_STATE", "device": "desktop-cuda", "from": "OFF",
                          "to": "RESTORED", "reason": "capability_probe", "at_us": 0})
        self.assertEqual(event["command"][-1], RGC)
        self.assertEqual(len(event["result"]), 4)
        with self.assertRaises(PhysicalAdapterError):
            controller.capability_probe()

    def test_refused_probe_makes_the_controller_unavailable_with_one_row_and_no_commands(self):
        for prefix in (("sudo", "-n", "-l", SMI, "-lgc"), ("sudo", "-n", "-l", TEE)):
            with self.subTest(prefix=prefix):
                self.run = RecordedRun()
                self.run.failures[prefix] = 1
                controller = self.controller(probe=False)
                self.assertFalse(controller.capability_probe())
                self.assertEqual(controller.state, "UNAVAILABLE")
                (event,) = controller.events
                self.assertEqual((event["from"], event["to"], event["reason"]), ("OFF", "UNAVAILABLE", "capability_probe_failed"))
                self.assertFalse(any(call[:3] == ["sudo", "-n", SMI] for call in self.run.calls))
                before = len(self.run.calls)
                controller.start()
                controller.note_next_arrival_us(1000 * S)
                controller.on_load_begin()
                controller.on_execution_start(gpu_command())
                controller.on_load_end()
                controller.end_trace()
                controller.close()
                self.assertEqual(self.run.calls[before:], [])
                self.assertEqual(len(controller.events), 1)
                self.assertIsNone(controller._thread)
        # an unreadable EPP is refused before any sudo command runs
        self.run = RecordedRun()
        for path in self.epp_paths:
            Path(path).write_text("")
        controller = self.controller(probe=False)
        self.assertFalse(controller.capability_probe())
        self.assertEqual(controller.events[0]["result"][-1]["stderr"], "CPU EPP is unreadable")

    def test_idle_enters_at_min_gap_and_restores_predictively_at_lead(self):
        controller = self.controller()
        self.run.calls.clear()
        controller._converge()  # arrival unknown
        self.assertEqual(controller.state, "RESTORED")
        controller.note_next_arrival_us(59 * S)  # gap 59 s < 60 s
        controller._converge()
        self.assertEqual((controller.state, self.run.calls), ("RESTORED", []))
        controller.note_next_arrival_us(120 * S)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        self.assertEqual(self.run.calls, [LGC_MIN, QUERY, ["sudo", "-n", TEE, *self.epp_paths]])
        self.assertEqual(self.epp_values(), ["power", "power"])
        event = controller.events[-1]
        self.assertEqual((event["from"], event["to"], event["reason"]), ("RESTORED", "IDLE_MIN", "idle_gap"))
        self.assertEqual(event["command"], [LGC_MIN, ["sudo", "-n", TEE, *self.epp_paths]])
        self.assertEqual(event["result"][0]["readback"]["clocks_sm_mhz"], 210)
        self.assertEqual(event["result"][1]["readback"], "power")
        # between lead and min_gap: hold
        self.run.calls.clear()
        self.advance(100)  # gap 20 s
        controller._converge()
        self.assertEqual((controller.state, self.run.calls), ("IDLE_MIN", []))
        self.advance(19.4)  # gap 0.6 s >= lead 0.5 s
        controller._converge()
        self.assertEqual((controller.state, self.run.calls), ("IDLE_MIN", []))
        self.assertAlmostEqual(controller._wake_timeout_s, 0.1, places=6)
        self.advance(0.2)  # gap 0.4 s < lead
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")
        self.assertEqual(self.run.calls, [RGC, QUERY, ["sudo", "-n", TEE, *self.epp_paths]])
        self.assertEqual(self.epp_values(), ["balance_performance", "balance_performance"])
        self.assertEqual(self.transitions(controller)[-1], ("IDLE_MIN", "RESTORED", "predictive_restore"))
        self.assertEqual(controller.events[-1]["at_us"], (self.now_ns[0] - self.epoch_ns) // 1000)
        # the arrival passed: gap negative, stay restored; no further arrivals -> idle again
        self.advance(5)
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")
        controller.note_next_arrival_us(None)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")

    def test_queued_tickets_and_transitions_keep_the_clocks_restored(self):
        controller = self.controller()
        controller.note_next_arrival_us(10_000 * S)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        controller.note_queued_start_us(3 * S)
        controller._converge()
        self.assertEqual(self.transitions(controller)[-1], ("IDLE_MIN", "RESTORED", "ticket_queued"))
        controller.note_queued_start_us(None)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        controller.note_transition_active(True)
        controller._converge()
        self.assertEqual(self.transitions(controller)[-1], ("IDLE_MIN", "RESTORED", "transition_active"))
        controller.note_transition_active(False)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        queued = [7 * S]
        controller.bind_queued_start_provider(lambda: queued[0])
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")
        queued[0] = None
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        for value in (-1, 1.0, True):
            with self.subTest(value=value), self.assertRaises(PhysicalAdapterError):
                controller.note_queued_start_us(value)
            with self.subTest(value=value, via="arrival"), self.assertRaises(PhysicalAdapterError):
                controller.note_next_arrival_us(value)
        controller.bind_queued_start_provider(lambda: 1.5)
        with self.assertRaises(PhysicalAdapterError):
            controller._converge()

    def test_execution_start_late_restores_synchronously_and_ignores_other_devices(self):
        controller = self.controller()
        controller.note_next_arrival_us(10_000 * S)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        before = len(self.run.calls)
        controller.on_execution_start(gpu_command(device_id="op15-phone"))
        self.assertEqual((controller.state, len(self.run.calls)), ("IDLE_MIN", before))
        command = gpu_command()
        controller.on_execution_start(command)
        self.assertEqual(controller.state, "RESTORED")
        self.assertEqual(self.run.calls[before:], [RGC, QUERY, ["sudo", "-n", TEE, *self.epp_paths]])
        self.assertEqual(self.transitions(controller)[-1], ("IDLE_MIN", "RESTORED", "late_restore"))
        controller._converge()  # active execution: stays restored
        self.assertEqual(controller.state, "RESTORED")
        controller.on_execution_finish(command)
        controller._converge()
        self.assertEqual(self.transitions(controller)[-1], ("RESTORED", "IDLE_MIN", "idle_gap"))
        # a second start while restored issues nothing synchronously
        controller.on_execution_start(command)
        before = len(self.run.calls)
        controller.on_execution_finish(command)
        self.assertEqual(len(self.run.calls), before)
        with self.assertRaises(PhysicalAdapterError):
            controller.on_execution_start(SimpleNamespace(ticket_id="x", executor_id="y", participants=()))

    def test_decode_cap_is_applied_by_the_thread_and_released_on_server_stop(self):
        controller = self.controller(policy(decode_cap={"sm_max_mhz": 1200}))
        self.assertEqual(self.run.calls[1], ["sudo", "-n", "-l", SMI, "-lgc", "210,1200", "-i", UUID])
        controller.note_next_arrival_us(10_000 * S)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        before = len(self.run.calls)
        first = gpu_command("a:attempt:0", "physical:desktop-gpu")
        controller.on_execution_start(first)
        self.assertEqual(controller.state, "DECODE_CAP")
        self.assertEqual(self.run.calls[before:], [["sudo", "-n", SMI, "-lgc", "210,1200", "-i", UUID], QUERY,
                                                   ["sudo", "-n", TEE, *self.epp_paths]])
        self.assertEqual(self.transitions(controller)[-1], ("IDLE_MIN", "DECODE_CAP", "late_restore"))
        controller.on_execution_finish(first)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        # from RESTORED the cap is not synchronous
        controller.note_queued_start_us(1)
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")
        before = len(self.run.calls)
        second = gpu_command("b:attempt:0", "physical:desktop-gpu")
        controller.on_execution_start(second)
        self.assertEqual((controller.state, len(self.run.calls)), ("RESTORED", before))
        controller._converge()
        self.assertEqual(controller.state, "DECODE_CAP")
        self.assertEqual(self.run.calls[before:], [["sudo", "-n", SMI, "-lgc", "210,1200", "-i", UUID], QUERY])
        controller.on_server_stopped("physical:other")
        controller._converge()
        self.assertEqual(controller.state, "DECODE_CAP")
        controller.on_server_stopped("physical:desktop-gpu")
        controller._converge()
        self.assertEqual(self.transitions(controller)[-1], ("DECODE_CAP", "RESTORED", "ticket_queued"))
        self.assertEqual(self.run.calls[-2], RGC)

    def test_load_min_locks_the_floor_around_a_transition_load(self):
        controller = self.controller(policy(load_min=True))
        controller.note_next_arrival_us(10_000 * S)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        before = len(self.run.calls)
        controller.on_load_begin()
        self.assertEqual(controller.state, "LOAD_MIN")
        # the clock is already at the floor: only the EPP goes back to its initial value
        self.assertEqual(self.run.calls[before:], [["sudo", "-n", TEE, *self.epp_paths]])
        self.assertEqual(self.epp_values(), ["balance_performance", "balance_performance"])
        self.assertEqual(self.transitions(controller)[-1], ("IDLE_MIN", "LOAD_MIN", "load_begin"))
        controller._converge()
        self.assertEqual(controller.state, "LOAD_MIN")
        before = len(self.run.calls)
        controller.on_load_end()
        self.assertEqual(self.transitions(controller)[-1], ("LOAD_MIN", "IDLE_MIN", "load_end"))
        self.assertEqual(self.run.calls[before:], [["sudo", "-n", TEE, *self.epp_paths]])
        count = len(controller.events)
        controller.on_load_end()
        self.assertEqual(len(controller.events), count)
        # from RESTORED with a transition active the load locks the floor and the end restores
        controller.note_transition_active(True)
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")
        before = len(self.run.calls)
        controller.on_load_begin()
        self.assertEqual(self.run.calls[before:], [LGC_MIN, QUERY])
        controller.on_load_end()
        self.assertEqual(self.run.calls[before + 2:], [RGC, QUERY])
        self.assertEqual(self.transitions(controller)[-1], ("LOAD_MIN", "RESTORED", "load_end"))
        # a GPU execution during a load restores at once
        controller.on_load_begin()
        controller.on_execution_start(gpu_command())
        self.assertEqual(self.transitions(controller)[-1], ("LOAD_MIN", "RESTORED", "late_restore"))
        # without load_min a load begin from IDLE_MIN restores
        plain = self.controller()
        plain.note_next_arrival_us(10_000 * S)
        plain._converge()
        plain.on_load_begin()
        self.assertEqual(self.transitions(plain)[-1], ("IDLE_MIN", "RESTORED", "load_begin"))

    def test_first_failed_command_makes_the_controller_unavailable_without_further_commands(self):
        for outcome in (1, subprocess.TimeoutExpired(SMI, 5), OSError("gone")):
            with self.subTest(outcome=outcome):
                self.run = RecordedRun()
                controller = self.controller()
                self.run.failures[("sudo", "-n", SMI, "-lgc")] = outcome
                controller.note_next_arrival_us(10_000 * S)
                controller._converge()
                self.assertEqual(controller.state, "UNAVAILABLE")
                event = controller.events[-1]
                self.assertEqual((event["from"], event["to"], event["reason"]), ("RESTORED", "UNAVAILABLE", "command_failed"))
                self.assertEqual(event["command"], [LGC_MIN])
                self.assertEqual(self.epp_values(), ["balance_performance", "balance_performance"])
                before = len(self.run.calls)
                controller._converge()
                controller.on_execution_start(gpu_command())
                controller.on_load_begin()
                controller.on_load_end()
                controller.end_trace()
                controller.close()
                self.assertEqual(self.run.calls[before:], [])
                self.assertEqual(len(controller.events), 2)

    def test_close_after_a_later_failure_still_lifts_a_lock_that_succeeded(self):
        controller = self.controller()
        controller.note_next_arrival_us(10_000 * S)
        self.run.failures[("sudo", "-n", TEE)] = subprocess.TimeoutExpired(TEE, 5)
        controller._converge()
        self.assertEqual(controller.state, "UNAVAILABLE")
        self.assertEqual(controller.events[-1]["command"], [LGC_MIN, ["sudo", "-n", TEE, *self.epp_paths]])
        before = len(self.run.calls)
        controller.close()
        self.assertEqual(self.run.calls[before:], [RGC, QUERY])
        event = controller.events[-1]
        self.assertEqual((event["from"], event["to"], event["reason"]), ("UNAVAILABLE", "UNAVAILABLE", "close_best_effort"))
        self.assertEqual(controller.state, "UNAVAILABLE")
        controller.close()
        self.assertEqual(len(self.run.calls), before + 2)

    def test_close_and_end_trace_restore_and_the_events_file_tracks_every_event(self):
        controller = self.controller()
        controller.note_next_arrival_us(10_000 * S)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        controller.end_trace()
        self.assertEqual(self.transitions(controller)[-1], ("IDLE_MIN", "RESTORED", "end_trace"))
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")  # frozen: never idle again
        controller.close()
        self.assertEqual(controller.state, "RESTORED")
        self.assertEqual(self.epp_values(), ["balance_performance", "balance_performance"])
        persisted = json.loads((self.output / "DEVICE_POWER_EVENTS.json").read_text())
        self.assertEqual(persisted, list(controller.events))
        events = controller.events
        events[0]["result"][0]["command"].append("mutated")
        self.assertNotIn("mutated", controller.events[0]["result"][0]["command"])
        # close from IDLE_MIN restores with the close reason
        other = self.controller()
        other.note_next_arrival_us(None)
        other._converge()
        self.assertEqual(other.state, "IDLE_MIN")
        other.close()
        self.assertEqual(self.transitions(other)[-1], ("IDLE_MIN", "RESTORED", "close"))
        self.assertEqual(self.run.calls[-3:], [RGC, QUERY, ["sudo", "-n", TEE, *self.epp_paths]])

    def test_thread_converges_and_stops(self):
        controller = self.controller(tick_interval_s=0.01)
        controller.start()
        controller.note_next_arrival_us(None)
        deadline = time.monotonic() + 5
        while controller.state != "IDLE_MIN" and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(controller.state, "IDLE_MIN")
        thread = controller._thread
        self.assertTrue(thread.is_alive() and thread.daemon)
        controller.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(controller.state, "RESTORED")
        self.assertEqual(threading.active_count(), threading.active_count())

    def test_setup_is_typed(self):
        for changes in ({"policy": POLICY}, {"epoch_ns_provider": 5}, {"output_directory": str(self.output)},
                        {"tick_interval_s": 0}, {"tick_interval_s": "1"}):
            options = {"policy": policy(), "epoch_ns_provider": lambda: 0, "output_directory": self.output, **changes}
            with self.subTest(changes=changes), self.assertRaises(PhysicalAdapterError):
                DevicePowerController(options.pop("policy"), run=self.run, **options)
        with self.assertRaises(PhysicalAdapterError):
            NvidiaClockControl("", run=self.run)


class RigAndRunnerWiringTests(ControllerHarness):
    def test_rig_configuration_revalidates_the_policy_against_the_gpu_device(self):
        self.assertIsNone(rig_device_power_policy(None, "desktop-cuda"))
        self.assertEqual(rig_device_power_policy(POLICY, "desktop-cuda"), policy())
        self.assertEqual(rig_device_power_policy(policy(), "desktop-cuda"), policy())
        for value, device in ((POLICY, "desktop-cpu"), ({**POLICY, "load_min": 1}, "desktop-cuda"),
                              ("desktop-cuda", "desktop-cuda"), ([POLICY], "desktop-cuda")):
            with self.subTest(value=value, device=device), self.assertRaises(PhysicalAdapterError):
                rig_device_power_policy(value, device)

    def test_result_key_appears_only_with_a_policy(self):
        self.assertEqual(runner._device_power_result(SimpleNamespace()), {})
        self.assertEqual(runner._device_power_result(SimpleNamespace(configuration=SimpleNamespace(device_power=None))), {})
        row = {"kind": "DEVICE_POWER_STATE", "device": "desktop-cuda", "from": "OFF", "to": "UNAVAILABLE",
               "reason": "capability_probe_failed", "at_us": 0, "command": [], "result": []}
        rig = SimpleNamespace(configuration=SimpleNamespace(device_power=policy()), device_power_events=(row,))
        self.assertEqual(runner._device_power_result(rig), {"device_power_events": [row]})
        empty = SimpleNamespace(configuration=SimpleNamespace(device_power=policy()), device_power_events=())
        self.assertEqual(runner._device_power_result(empty), {"device_power_events": []})
        with self.assertRaises(Exception):
            runner._device_power_result(replace_events(rig, ({"kind": "OTHER"},)))
        # the runner feeds the next arrival before waiting and closes the sequence after the loop
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("_note_next_arrival(rig, request.arrival_us)\n        coordinator.wait_for_arrival(request.arrival_us)", source)
        self.assertIn("    _note_next_arrival(rig, None)\n", source)
        self.assertIn("**_device_power_result(rig),", source)
        self.assertIn("default_host_metric_callbacks(gpu_clocks=device_power is not None)", source)

    def test_scheduler_reports_no_queued_start_when_nothing_is_queued(self):
        from research_dev.scheduler import UnifiedScheduler
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        self.assertIsNone(scheduler.runtime_queued_start_us())

    def test_rig_hooks_are_wired_outside_the_rig_lock(self):
        from research_dev.scheduler.adapters import heterogeneous_rig
        from research_dev.scheduler.adapters.heterogeneous_rig_ops import lifecycle, transitions
        rig_source = Path(heterogeneous_rig.__file__).read_text(encoding="utf-8")
        for snippet in ("power.on_execution_start(command)", "power.on_execution_finish(command)",
                        "power.on_load_begin()", "power.on_load_end()", "controller.end_trace()",
                        "self._device_power.capability_probe()", "power.bind_queued_start_provider(queued_start)"):
            self.assertIn(snippet, rig_source)
        self.assertIn("power.on_server_stopped(executor_id)", Path(lifecycle.__file__).read_text(encoding="utf-8"))
        lifecycle_source = Path(lifecycle.__file__).read_text(encoding="utf-8")
        self.assertLess(lifecycle_source.index("power.close()"), lifecycle_source.index("self._sampler.stop()"))
        self.assertIn("power.note_transition_active(True)", Path(transitions.__file__).read_text(encoding="utf-8"))


def replace_events(rig, events):
    return SimpleNamespace(configuration=rig.configuration, device_power_events=events)


class HostSamplerAndPreflightTests(unittest.TestCase):
    def test_gpu_sample_carries_clock_columns_only_when_requested(self):
        base = "NVIDIA GeForce RTX 4060 Ti, GPU-1, 16380, 2000, 14380, 8, 27.61"
        with mock.patch.object(host_runtime.subprocess, "run", return_value=SimpleNamespace(stdout=base + "\n")) as run:
            plain = host_runtime.nvidia_gpu_snapshot()
        self.assertNotIn("clocks.sm", run.call_args.args[0][1])
        self.assertEqual(set(plain), {"memory_free_bytes", "memory_total_bytes", "memory_used_bytes", "name",
                                      "power_mw", "utilization_pct", "uuid"})
        with mock.patch.object(host_runtime.subprocess, "run", return_value=SimpleNamespace(stdout=base + ", 210, 405, P8\n")) as run:
            clocked = host_runtime.default_host_metric_callbacks(gpu_clocks=True).gpu_snapshot()
        self.assertTrue(run.call_args.args[0][1].endswith("power.draw,clocks.sm,clocks.mem,pstate"))
        self.assertEqual({key: clocked[key] for key in ("clocks_sm_mhz", "clocks_mem_mhz", "pstate")},
                         {"clocks_sm_mhz": 210, "clocks_mem_mhz": 405, "pstate": "P8"})
        self.assertEqual({key: value for key, value in clocked.items() if key in plain}, plain)
        self.assertIs(host_runtime.default_host_metric_callbacks().gpu_snapshot, host_runtime.nvidia_gpu_snapshot)
        with mock.patch.object(host_runtime.subprocess, "run", return_value=SimpleNamespace(stdout=base + "\n")):
            with self.assertRaises(PhysicalAdapterError):
                host_runtime.nvidia_gpu_snapshot(clocks=True)
        for flag in (1, None):
            with self.subTest(flag=flag), self.assertRaises(PhysicalAdapterError):
                host_runtime.nvidia_gpu_snapshot(clocks=flag)
            with self.subTest(flag=flag, via="callbacks"), self.assertRaises(PhysicalAdapterError):
                host_runtime.default_host_metric_callbacks(gpu_clocks=flag)

    def test_preflight_check_is_advisory_and_absent_without_the_policy(self):
        self.assertEqual(preflight._device_power_checks(argparse.Namespace(device_power=None)), [])
        self.assertEqual(preflight._device_power_checks(argparse.Namespace()), [])
        args = argparse.Namespace(device_power=policy().to_json())
        for available, status in ((True, "PASS"), (False, "WARN")):
            with mock.patch.object(preflight, "device_power_capability", return_value=(available, "detail", ())):
                (row,) = preflight._device_power_checks(args)
            self.assertEqual(row.to_json(), {"check_id": "device-power-control:desktop-cuda", "detail": "detail", "status": status})

    def test_capability_helper_never_raises_and_names_the_refused_command(self):
        run = RecordedRun()
        root = Path(tempfile.mkdtemp(prefix="device-power-epp-"))
        self.addCleanup(shutil.rmtree, root, True)
        (root / "cpu0" / "cpufreq").mkdir(parents=True)
        (root / "cpu0" / "cpufreq" / "energy_performance_preference").write_text("balance_performance\n")
        available, detail, rows = device_power_capability(policy(), run=run, cpu_root=root)
        self.assertTrue(available)
        self.assertEqual(detail, "sudo permits nvidia-smi -lgc/-rgc for " + UUID + " and tee EPP; EPP balance_performance")
        self.assertEqual([row["command"][:4] for row in rows], [["sudo", "-n", "-l", SMI]] * 2 + [["sudo", "-n", "-l", TEE]])
        run.failures[("sudo", "-n", "-l", SMI, "-rgc")] = FileNotFoundError("sudo")
        available, detail, _rows = device_power_capability(policy(), run=run, cpu_root=root)
        self.assertFalse(available)
        self.assertTrue(detail.startswith("refused: sudo -n -l " + SMI + " -rgc -> no exit FileNotFoundError"))
        self.assertTrue(detail.isascii())
        available, detail, _rows = device_power_capability(policy(idle={**IDLE, "cpu_epp": None}), run=RecordedRun(), cpu_root=root)
        self.assertTrue(available)
        self.assertEqual(detail, "sudo permits nvidia-smi -lgc/-rgc for " + UUID)


class AnalysisToolTests(unittest.TestCase):
    PAID_START_NS = 1_000_000_000_000
    PAID_S = 20

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="device-power-run-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def write_run(self, events, *, result_events=None):
        result = {"paid_start_ns": self.PAID_START_NS, "paid_end_ns": self.PAID_START_NS + self.PAID_S * 10 ** 9}
        if result_events is not None:
            result["device_power_events"] = result_events
        (self.root / "RESULT.json").write_text(json.dumps(result))
        if events is not None:
            (self.root / "DEVICE_POWER_EVENTS.json").write_text(json.dumps(events))
        rows = []
        energy_uj = 5_000_000_000
        step_ns = 500_000_000
        for index in range(-4, self.PAID_S * 2 + 5):
            t_ns = self.PAID_START_NS + index * step_ns
            seconds = index * 0.5
            power_mw = 30_000 if seconds < 10 else 11_000
            energy_uj += 20 * 500_000  # 20 W for 0.5 s in uJ
            rows.append({"gpu": {"power_mw": power_mw, "sample_t_ns": t_ns, "uuid": UUID, "utilization_pct": 0},
                         "rapl_package": {"energy_uj": energy_uj, "max_energy_range_uj": 262_143_328_850, "sample_t_ns": t_ns},
                         "t_ns": t_ns})
        rows.append({"gpu": {"power_mw": 1, "sample_t_ns": 0}, "rapl_package": None, "t_ns": 0})  # probe error row
        with (self.root / "resource-samples.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    @staticmethod
    def event(at_us, source, to, reason, readback):
        return {"kind": "DEVICE_POWER_STATE", "device": "desktop-cuda", "from": source, "to": to, "reason": reason,
                "at_us": at_us, "command": [["sudo", "-n", SMI, "-rgc"]],
                "result": [{"command": ["sudo", "-n", SMI, "-rgc"], "returncode": 0, "stderr": "", "stdout": "",
                            "readback": readback, "duration_us": 20_000}]}

    def test_energy_per_state_between_event_boundaries(self):
        events = [
            self.event(-2_000_000, "OFF", "RESTORED", "capability_probe", {"clocks_sm_mhz": 2505, "clocks_mem_mhz": 405, "pstate": "P0"}),
            self.event(10 * S, "RESTORED", "IDLE_MIN", "idle_gap", {"clocks_sm_mhz": 210, "clocks_mem_mhz": 405, "pstate": "P8"}),
            self.event(25 * S, "IDLE_MIN", "RESTORED", "close", None),  # after the paid window
        ]
        self.write_run(events)
        report = device_power_energy.analyze(self.root)
        self.assertEqual(report["event_count"], 3)
        self.assertTrue(report["policy_present"])
        self.assertEqual(sorted(report["per_state"]), ["IDLE_MIN", "RESTORED"])
        restored, idle = report["per_state"]["RESTORED"], report["per_state"]["IDLE_MIN"]
        self.assertAlmostEqual(restored["seconds"], 10.0)
        self.assertAlmostEqual(idle["seconds"], 10.0)
        self.assertAlmostEqual(restored["gpu_j"], 300.0, delta=10.0)
        self.assertAlmostEqual(idle["gpu_j"], 110.0, delta=10.0)
        self.assertAlmostEqual(restored["cpu_j"], 200.0, delta=0.5)
        self.assertAlmostEqual(idle["cpu_j"], 200.0, delta=0.5)
        self.assertAlmostEqual(restored["gpu_w"], 30.0, delta=1.0)
        self.assertAlmostEqual(idle["gpu_w"], 11.0, delta=1.0)
        self.assertEqual([(row["state"], row["start_ns"], row["end_ns"]) for row in report["segments"]], [
            ("RESTORED", self.PAID_START_NS, self.PAID_START_NS + 10 * 10 ** 9),
            ("IDLE_MIN", self.PAID_START_NS + 10 * 10 ** 9, self.PAID_START_NS + 20 * 10 ** 9),
        ])
        self.assertEqual([row["readback"] for row in report["readbacks"]][1], {"clocks_sm_mhz": 210, "clocks_mem_mhz": 405, "pstate": "P8"})
        text = device_power_energy.render(report)
        self.assertIn("IDLE_MIN", text)
        self.assertIn("readbacks", text)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(device_power_energy.main(["--run-dir", str(self.root), "--json"]), 0)
        self.assertEqual(json.loads(printed.getvalue())["event_count"], 3)
        # the RESULT copy is the fallback when the incremental file is absent
        (self.root / "DEVICE_POWER_EVENTS.json").unlink()
        self.write_run(None, result_events=events[:2])
        self.assertEqual(device_power_energy.analyze(self.root)["event_count"], 2)

    def test_run_without_the_policy_has_one_uncontrolled_state(self):
        self.write_run(None)
        report = device_power_energy.analyze(self.root)
        self.assertFalse(report["policy_present"])
        self.assertEqual(list(report["per_state"]), ["UNCONTROLLED"])
        total = report["per_state"]["UNCONTROLLED"]
        self.assertAlmostEqual(total["seconds"], 20.0)
        self.assertAlmostEqual(total["gpu_j"], 410.0, delta=10.0)
        self.assertAlmostEqual(total["cpu_j"], 400.0, delta=0.5)
        self.assertEqual(total["uncovered"], 0)
        self.assertEqual(report["readbacks"], [])

    def test_uncovered_segments_are_reported_not_integrated(self):
        self.write_run([self.event(0, "OFF", "RESTORED", "capability_probe", None)])
        (self.root / "resource-samples.jsonl").write_text("")
        report = device_power_energy.analyze(self.root)
        self.assertEqual(report["per_state"]["RESTORED"]["uncovered"], 1)
        self.assertEqual(report["per_state"]["RESTORED"]["seconds"], 0.0)
        self.assertIsNone(report["per_state"]["RESTORED"]["gpu_w"])
        self.assertIn("uncovered", report["segments"][0])
        (self.root / "DEVICE_POWER_EVENTS.json").write_text(json.dumps([{"kind": "OTHER", "at_us": 0}]))
        with self.assertRaises(ValueError):
            device_power_energy.analyze(self.root)


if __name__ == "__main__":
    unittest.main()
