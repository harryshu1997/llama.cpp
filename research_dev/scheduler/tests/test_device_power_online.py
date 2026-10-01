#!/usr/bin/env python3
"""Online device power control (``device_power.arrival_information``) and prefill protection.

The legacy controller is fed the next trace arrival before it happens (an oracle). With
``arrival_information: "online"`` the runner passes only arrivals that already happened, the
controller refuses the calendar, and an ``ArrivalGapPredictor`` learned from observed gaps plus a
ski-rental timeout decide the idle drop. ``decode_cap.protect_prefill`` keeps full clocks until
every active execution produced its first token. An explicit ``arrival_information`` adds
telemetry (idle intervals, execution-start waits) that the analysis tool turns into prediction
quality. Without the new keys every event and RESULT stays byte-identical.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler.adapters import PhysicalAdapterError
from research_dev.scheduler.adapters.device_power import (
    ONLINE_IDLE_SETTLE_US,
    ArrivalGapPredictor,
    DevicePowerController,
)
from research_dev.scheduler.campaigns.burstgpt import arguments, runner
from research_dev.scheduler.campaigns.burstgpt.tools import device_power_energy, device_power_replay
from research_dev.scheduler.config import (
    CampaignManifest,
    DevicePowerConfiguration,
    SchedulerConfigurationError,
)

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from test_device_power import (  # noqa: E402
    IDLE,
    LGC_MIN,
    POLICY,
    QUERY,
    RGC,
    SMI,
    UUID,
    ControllerHarness,
    gpu_command,
)
import test_device_power  # noqa: E402  (module access: its test classes are not collected twice)

S = 1_000_000
IDLE_NO_EPP = {**IDLE, "cpu_epp": None, "min_gap_s": 2}
ONLINE_IDLE = {"predictor": "global", "max_arrival_probability_ppm": 250_000, "fallback_idle_ms": 2_000}
ONLINE = {"device": "desktop-cuda", "gpu_uuid": UUID, "idle": IDLE_NO_EPP, "arrival_information": "online",
          "online_idle": ONLINE_IDLE}
CAP = ["sudo", "-n", SMI, "-lgc", "210,1200", "-i", UUID]


def configuration(row) -> DevicePowerConfiguration:
    return DevicePowerConfiguration.from_json(row)


class OnlineConfigurationTests(unittest.TestCase):
    def test_legacy_policies_serialize_exactly_as_before(self):
        legacy = {**POLICY, "decode_cap": {"sm_max_mhz": 1200}, "load_min": True}
        for row in (POLICY, legacy, {**legacy, "decode_cap": {"sm_max_mhz": 1200, "protect_prefill": False}}):
            with self.subTest(row=row):
                parsed = configuration(row)
                self.assertIsNone(parsed.arrival_information)
                self.assertIsNone(parsed.online_idle)
                self.assertFalse(parsed.online)
                self.assertFalse(parsed.decode_cap is not None and parsed.decode_cap.protect_prefill)
                self.assertNotIn("arrival_information", parsed.to_json())
                self.assertNotIn("online_idle", parsed.to_json())
        self.assertEqual(json.dumps(configuration(legacy).to_json(), sort_keys=True), json.dumps(
            {"decode_cap": {"sm_max_mhz": 1200}, "device": "desktop-cuda", "gpu_uuid": UUID, "idle": IDLE,
             "load_min": True}, sort_keys=True))
        # the positional constructor of the legacy fields is unchanged
        self.assertEqual(DevicePowerConfiguration("desktop-cuda", UUID, configuration(POLICY).idle), configuration(POLICY))

    def test_online_oracle_and_protect_prefill_round_trip(self):
        online = configuration({**ONLINE, "decode_cap": {"sm_max_mhz": 1200, "protect_prefill": True}})
        self.assertTrue(online.online)
        self.assertEqual(online.online_idle.to_json(), ONLINE_IDLE)
        self.assertEqual(online.decode_cap.to_json(), {"sm_max_mhz": 1200, "protect_prefill": True})
        self.assertEqual(configuration(online.to_json()), online)
        oracle = configuration({**POLICY, "arrival_information": "oracle"})
        self.assertFalse(oracle.online)
        self.assertEqual(oracle.to_json()["arrival_information"], "oracle")
        self.assertEqual(configuration(oracle.to_json()), oracle)
        self.assertEqual(configuration({**POLICY, "arrival_information": None}), configuration(POLICY))
        per_model = configuration({**ONLINE, "online_idle": {**ONLINE_IDLE, "predictor": "per_model",
                                                              "fallback_idle_ms": None}})
        self.assertIsNone(per_model.online_idle.fallback_idle_ms)
        self.assertEqual(configuration(per_model.to_json()), per_model)
        self.assertEqual(arguments.device_power_json(json.dumps(online.to_json())), online.to_json())
        manifest = CampaignManifest.from_json({**test_device_power.ManifestAndArgumentTests.ROW, "device_power": online.to_json()},
                                              Path("/inputs"))
        self.assertEqual(manifest.device_power, online)
        self.assertEqual(CampaignManifest.from_json(manifest.to_json(), Path("/inputs")), manifest)

    def test_rejects_inconsistent_online_rows(self):
        invalid = (
            {**ONLINE, "online_idle": None},                               # online without its rule
            {**POLICY, "online_idle": ONLINE_IDLE},                        # rule without online
            {**POLICY, "arrival_information": "oracle", "online_idle": ONLINE_IDLE},
            {"device": "desktop-cuda", "gpu_uuid": UUID, "decode_cap": {"sm_max_mhz": 1200}},
            {**{key: value for key, value in ONLINE.items() if key != "idle"}, "load_min": True},
            {**POLICY, "arrival_information": "future"},
            {**POLICY, "arrival_information": True},
            {**ONLINE, "online_idle": {**ONLINE_IDLE, "predictor": "hazard"}},
            {**ONLINE, "online_idle": {**ONLINE_IDLE, "max_arrival_probability_ppm": 0}},
            {**ONLINE, "online_idle": {**ONLINE_IDLE, "max_arrival_probability_ppm": 1_000_000}},
            {**ONLINE, "online_idle": {**ONLINE_IDLE, "max_arrival_probability_ppm": 0.25}},
            {**ONLINE, "online_idle": {**ONLINE_IDLE, "max_arrival_probability_ppm": True}},
            {**ONLINE, "online_idle": {**ONLINE_IDLE, "fallback_idle_ms": 0}},
            {**ONLINE, "online_idle": {**ONLINE_IDLE, "fallback_idle_ms": 1.5}},
            {**ONLINE, "online_idle": {**ONLINE_IDLE, "extra": 1}},
            {**ONLINE, "online_idle": {"predictor": "global"}},
            {**ONLINE, "online_idle": []},
            {**POLICY, "decode_cap": {"sm_max_mhz": 1200, "protect_prefill": 1}},
            {**POLICY, "decode_cap": {"sm_max_mhz": 1200, "protect_prefill": "yes"}},
            {**POLICY, "decode_cap": {"protect_prefill": True}},
        )
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(SchedulerConfigurationError):
                configuration(row)


class ArrivalGapPredictorTests(unittest.TestCase):
    def test_pessimistic_conditional_estimate_from_observed_gaps(self):
        predictor = ArrivalGapPredictor(2 * S, per_model=False)
        self.assertIsNone(predictor.estimate(0))
        predictor.observe("qwen", 0)
        self.assertEqual(predictor.estimate(10 * S)["probability_ppm"], 1_000_000)  # no gap yet
        for at_s in (100, 200, 300, 400):
            predictor.observe("gemma" if at_s % 200 else "qwen", at_s * S)
        estimate = predictor.estimate(450 * S)  # 50 s after the last arrival: 4 gaps of 100 s survive
        self.assertEqual(estimate["sequences"], [{"key": "*", "elapsed_us": 50 * S, "gaps": 4, "at_risk": 4, "within": 0}])
        self.assertEqual(estimate["probability_ppm"], 200_000)
        self.assertEqual(predictor.estimate(499 * S)["probability_ppm"], 1_000_000)  # every gap ends within 2 s
        self.assertEqual(predictor.estimate(501 * S)["probability_ppm"], 1_000_000)  # beyond every gap: no evidence
        with self.assertRaises(PhysicalAdapterError):
            predictor.observe("qwen", 399 * S)
        for arguments_ in ((0,), (-1,), (1.5,)):
            with self.subTest(horizon=arguments_), self.assertRaises(PhysicalAdapterError):
                ArrivalGapPredictor(*arguments_, per_model=False)
        with self.assertRaises(PhysicalAdapterError):
            ArrivalGapPredictor(S, per_model=1)

    def test_per_model_sequences_combine(self):
        predictor = ArrivalGapPredictor(2 * S, per_model=True)
        for model, at_s in (("a", 0), ("b", 5), ("a", 100), ("b", 105), ("a", 200), ("b", 205)):
            predictor.observe(model, at_s * S)
        estimate = predictor.estimate(250 * S)  # each model: 2 gaps of 100 s survive, none ends within 2 s
        self.assertEqual([row["key"] for row in estimate["sequences"]], ["a", "b"])
        self.assertEqual([row["at_risk"] for row in estimate["sequences"]], [2, 2])
        # 1 - (2/3)^2 = 5/9
        self.assertEqual(estimate["probability_ppm"], 555_556)


class OnlineControllerTests(ControllerHarness):
    def online(self, **changes) -> DevicePowerController:
        return self.controller(configuration({**ONLINE, **changes}))

    def telemetry_file(self):
        return json.loads((self.output / "DEVICE_POWER_TELEMETRY.json").read_text())

    def test_online_refuses_the_calendar_and_future_observations(self):
        controller = self.online()
        for value in (5 * S, None):
            with self.subTest(value=value), self.assertRaises(PhysicalAdapterError):
                controller.note_next_arrival_us(value)
        with self.assertRaises(PhysicalAdapterError):
            controller.note_arrival_observed("r0", "qwen", 1)  # now is 0 on the RESULT clock
        self.advance(3)
        controller.note_arrival_observed("r0", "qwen", 3 * S)
        with self.assertRaises(PhysicalAdapterError):
            controller.note_arrival_observed("r1", "qwen", 2 * S)  # out of order
        for row in (("", "qwen", 3 * S), ("r1", "", 3 * S), ("r1", "qwen", 3.0), ("r1", "qwen", True), ("r1", "qwen", -1)):
            with self.subTest(row=row), self.assertRaises(PhysicalAdapterError):
                controller.note_arrival_observed(*row)
        self.assertIsNone(controller._inputs.next_arrival_us)
        self.assertFalse(controller._inputs.arrivals_finished)
        legacy = self.controller()
        with self.assertRaises(PhysicalAdapterError):
            legacy.note_arrival_observed("r0", "qwen", 0)
        self.assertIsNone(legacy.telemetry)

    def test_a_legacy_policy_writes_no_telemetry(self):
        controller = self.controller(configuration({**POLICY, "decode_cap": {"sm_max_mhz": 1200}}))
        controller.note_next_arrival_us(100 * S)
        controller._converge()
        command = gpu_command()
        controller.on_execution_start(command)
        controller.note_first_token(command.request_id)
        controller.on_execution_finish(command)
        controller.end_trace()
        controller.close()
        self.assertIsNone(controller.telemetry)
        self.assertFalse((self.output / "DEVICE_POWER_TELEMETRY.json").exists())
        self.assertFalse(any("decision" in row for row in controller.events))

    def test_no_drop_before_the_first_observed_arrival_then_hold_fallback_and_restore(self):
        controller = self.online()
        self.run.calls.clear()
        self.advance(100)
        controller._converge()
        self.assertEqual((controller.state, self.run.calls), ("RESTORED", []))
        controller.note_arrival_observed("r0", "qwen", 100 * S)
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")  # arrival observed, its work has not started
        self.advance(1.9)
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")
        # the work starts and finishes; the GPU is idle again
        command = gpu_command("r0:attempt:0")
        controller.on_execution_start(command)
        self.advance(30)
        controller.on_execution_finish(command)
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")  # one arrival, no gap: only the fallback may drop
        self.assertAlmostEqual(controller._wake_timeout_s, 0.5)
        self.advance(1.99)
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")
        self.assertAlmostEqual(controller._wake_timeout_s, 0.01, places=5)
        self.advance(0.01)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        event = controller.events[-1]
        self.assertEqual((event["from"], event["to"], event["reason"]), ("RESTORED", "IDLE_MIN", "idle_timeout"))
        self.assertEqual(event["command"], [LGC_MIN])
        self.assertEqual(event["decision"]["idle_us"], 2 * S)
        self.assertEqual(event["decision"]["estimate"]["probability_ppm"], 1_000_000)
        # an observed arrival restores on the thread's next convergence
        self.advance(10)
        controller.note_arrival_observed("r1", "gemma", controller._at_us())
        controller._converge()
        self.assertEqual(self.transitions(controller)[-1], ("IDLE_MIN", "RESTORED", "arrival_observed"))
        self.assertEqual(self.run.calls[-2:], [RGC, QUERY])
        # the arrival's work never reaches the GPU (a CPU-only request): the hold expires
        self.advance(2)
        controller._converge()
        self.assertEqual(self.transitions(controller)[-1], ("RESTORED", "IDLE_MIN", "idle_timeout"))
        telemetry = controller.telemetry
        self.assertEqual([row["request_id"] for row in telemetry["observed_arrivals"]], ["r0", "r1"])
        self.assertEqual(telemetry["idle_intervals"][0]["end_reason"], "execution_active")
        self.assertEqual(telemetry["idle_intervals"][-1]["end_reason"], "open")
        self.assertEqual(self.telemetry_file()["observed_arrivals"], telemetry["observed_arrivals"])

    def test_predictor_drops_after_the_settle_time_and_vetoes_when_an_arrival_is_likely(self):
        controller = self.online(online_idle={**ONLINE_IDLE, "fallback_idle_ms": 60_000})
        for at_s in (0, 100, 200, 300, 400):
            self.now_ns[0] = self.epoch_ns + at_s * S * 1000
            controller.note_arrival_observed("r%d" % at_s, "qwen", at_s * S)
            command = gpu_command("r%d:attempt:0" % at_s)
            controller.on_execution_start(command)
            self.advance(40)
            controller.on_execution_finish(command)
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")  # settling
        self.assertAlmostEqual(controller._wake_timeout_s, 0.5)
        self.advance(ONLINE_IDLE_SETTLE_US / 1e6)
        controller._converge()
        event = controller.events[-1]
        self.assertEqual((event["to"], event["reason"]), ("IDLE_MIN", "predicted_idle"))
        self.assertEqual(event["decision"]["estimate"]["probability_ppm"], 200_000)  # 4 gaps of 100 s at 41 s
        # IDLE_MIN holds until an input restores it: the predictor never restores by itself
        self.advance(55)
        controller._converge()
        self.assertEqual((controller.state, self.transitions(controller)[-1][2]), ("IDLE_MIN", "predicted_idle"))
        # short regular gaps: an arrival within the horizon is likely -> no predicted drop
        other = self.online(online_idle={**ONLINE_IDLE, "fallback_idle_ms": 60_000})
        start_s = (self.now_ns[0] - self.epoch_ns) // 10 ** 9 + 1
        for index in range(5):
            self.now_ns[0] = self.epoch_ns + (start_s + 3 * index) * 10 ** 9
            other.note_arrival_observed("s%d" % index, "qwen", (start_s + 3 * index) * S)
            command = gpu_command("s%d:attempt:0" % index)
            other.on_execution_start(command)
            self.advance(0.5)
            other.on_execution_finish(command)
        self.advance(1.0)
        other._converge()
        self.assertEqual(other.state, "RESTORED")
        self.assertEqual(other._decision["estimate"]["probability_ppm"], 1_000_000)

    def test_queue_transition_load_and_execution_keep_or_restore_full_clocks(self):
        controller = self.online(load_min=True)
        controller.note_arrival_observed("r0", "qwen", 0)
        self.advance(5)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")  # hold expired, fallback reached
        controller.note_queued_start_us(6 * S)
        controller._converge()
        self.assertEqual(self.transitions(controller)[-1], ("IDLE_MIN", "RESTORED", "ticket_queued"))
        controller.note_queued_start_us(None)
        controller.note_transition_active(True)
        controller.on_load_begin()
        self.assertEqual(controller.state, "LOAD_MIN")
        controller.on_load_end()
        controller.note_transition_active(False)
        self.assertEqual(controller.state, "RESTORED")
        self.advance(3)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        before = len(self.run.calls)
        command = gpu_command("r0:attempt:0")
        controller.on_execution_start(command)  # synchronous late restore, charged to the start
        self.assertEqual(self.run.calls[before:], [RGC, QUERY])
        self.assertEqual(self.transitions(controller)[-1], ("IDLE_MIN", "RESTORED", "late_restore"))
        wait = controller.telemetry["execution_waits"][-1]
        self.assertEqual({key: wait[key] for key in ("ticket_id", "request_id", "state_before", "state_after", "synchronous")},
                         {"ticket_id": "r0:attempt:0", "request_id": command.request_id, "state_before": "IDLE_MIN",
                          "state_after": "RESTORED", "synchronous": True})
        reasons = [row["end_reason"] for row in controller.telemetry["idle_intervals"]]
        self.assertEqual(reasons, ["ticket_queued", "transition_active", "execution_active"])
        controller.end_trace()
        self.assertEqual(self.telemetry_file()["idle_intervals"], controller.telemetry["idle_intervals"])

    def test_explicit_oracle_keeps_the_calendar_behaviour_and_adds_telemetry(self):
        legacy = self.controller(configuration({**POLICY, "idle": IDLE_NO_EPP}))
        oracle = self.controller(configuration({**POLICY, "idle": IDLE_NO_EPP, "arrival_information": "oracle"}))
        for controller in (legacy, oracle):
            controller.note_next_arrival_us(controller._at_us() + 100 * S)
            controller._converge()
            self.advance(99.4)
            controller._converge()
            command = gpu_command()
            controller.on_execution_start(command)
            controller.on_execution_finish(command)
            controller.note_next_arrival_us(None)
            controller._converge()
            controller.end_trace()
        strip = lambda rows: [{key: row[key] for key in ("from", "to", "reason", "command")} for row in rows]  # noqa: E731
        self.assertEqual(strip(legacy.events), strip(oracle.events))
        self.assertFalse(any("decision" in row for row in oracle.events))
        self.assertIsNone(legacy.telemetry)
        self.assertEqual(oracle.telemetry["arrival_information"], "oracle")
        self.assertEqual(len(oracle.telemetry["execution_waits"]), 1)


class PrefillProtectionTests(ControllerHarness):
    CAPPED = {**POLICY, "idle": IDLE_NO_EPP, "decode_cap": {"sm_max_mhz": 1200, "protect_prefill": True}}

    def test_cap_only_after_every_first_token_and_synchronous_restore_for_a_joiner(self):
        controller = self.controller(configuration(self.CAPPED))
        controller.note_next_arrival_us(10_000 * S)
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        first = gpu_command("a:attempt:0")
        controller.on_execution_start(first)
        self.assertEqual(self.transitions(controller)[-1], ("IDLE_MIN", "RESTORED", "late_restore"))
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")  # prompt processing at full clocks
        controller.note_first_token("unrelated")
        controller._converge()
        self.assertEqual(controller.state, "RESTORED")
        controller.note_first_token(first.request_id)
        controller._converge()
        self.assertEqual(self.transitions(controller)[-1], ("RESTORED", "DECODE_CAP", "execution_active"))
        self.assertEqual(self.run.calls[-2:], [CAP, QUERY])
        joiner = replace(gpu_command("b:attempt:0"), request_id="joiner")
        before = len(self.run.calls)
        controller.on_execution_start(joiner)  # the cap is lifted before the joiner's prompt
        self.assertEqual(self.run.calls[before:], [RGC, QUERY])
        self.assertEqual(self.transitions(controller)[-1], ("DECODE_CAP", "RESTORED", "prefill_restore"))
        controller.note_first_token("joiner")
        controller._converge()
        self.assertEqual(controller.state, "DECODE_CAP")
        # a retried attempt of a request starts in prefill again
        controller.on_execution_finish(first)
        retry = gpu_command("a:attempt:1")
        controller.on_execution_start(retry)
        self.assertEqual(self.transitions(controller)[-1], ("DECODE_CAP", "RESTORED", "prefill_restore"))
        controller.note_first_token(retry.request_id)
        controller._converge()
        self.assertEqual(controller.state, "DECODE_CAP")
        controller.on_server_stopped("physical:desktop-gpu")
        controller._converge()
        self.assertEqual(controller.state, "IDLE_MIN")
        self.assertEqual(controller._inputs.request_by_ticket, {})
        self.assertEqual(controller._inputs.decoding, set())

    def test_without_the_flag_the_first_token_is_ignored_and_the_cap_starts_at_once(self):
        controller = self.controller(configuration({**POLICY, "decode_cap": {"sm_max_mhz": 1200}}))
        command = gpu_command()
        controller.on_execution_start(command)
        controller._converge()
        self.assertEqual(controller.state, "DECODE_CAP")
        controller.note_first_token(command.request_id)
        self.assertEqual(controller._inputs.request_by_ticket, {})
        with self.assertRaises(PhysicalAdapterError):
            controller.note_first_token("")


class _FakeCoordinator:
    """``wait_for_arrival`` advances a shared fake clock to the arrival; ``submit`` records what
    the controller knew at that moment."""

    def __init__(self, harness, controller):
        self.harness = harness
        self.controller = controller
        self.knowledge = []

    def now_us(self):
        return (self.harness.now_ns[0] - self.harness.epoch_ns) // 1000

    def wait_for_arrival(self, arrival_us):
        self.harness.now_ns[0] = max(self.harness.now_ns[0], self.harness.epoch_ns + arrival_us * 1000)

    def observed_at_us(self):
        return self.now_us()

    def submit(self, submission, *, observed_at_us):
        inputs = self.controller._inputs
        telemetry = self.controller.telemetry
        self.knowledge.append({"now_us": self.now_us(), "request_id": submission.request.request_id,
                               "next_arrival_us": inputs.next_arrival_us, "finished": inputs.arrivals_finished,
                               "observed": [] if telemetry is None else [
                                   row["observed_at_us"] for row in telemetry["observed_arrivals"]]})
        self.harness.advance(1.0)  # scheduler submit time
        return SimpleNamespace(ticket_id=submission.request.request_id)

    def rejections(self):
        return ()


class _SpyRig:
    """The rig surface the runner uses for device power, forwarding to a real controller."""

    def __init__(self, coordinator_clock, controller):
        self.configuration = SimpleNamespace(device_power=controller.policy)
        self._clock = coordinator_clock
        self._controller = controller
        self.calls = []

    def note_next_arrival(self, arrival_us):
        self.calls.append(("next_arrival", arrival_us, self._clock()))
        self._controller.note_next_arrival_us(arrival_us)

    def note_arrival_observed(self, request_id, model_id, observed_at_us):
        self.calls.append(("arrival_observed", (request_id, model_id, observed_at_us), self._clock()))
        self._controller.note_arrival_observed(request_id, model_id, observed_at_us)

    def note_first_token(self, request_id):
        self.calls.append(("first_token", request_id, self._clock()))
        self._controller.note_first_token(request_id)

    def snapshot(self, request, model_id, captured_at_us):
        return SimpleNamespace(to_json=lambda: {"snapshot": request.request_id})


def _trace_row(event_id, index, arrival_s, model_id="qwen-model"):
    return {"combined_index": index, "model_id": model_id,
            "row": {"event_id": event_id, "arrival_us": arrival_s * S, "slo_us": 600 * S, "input_tokens": 4,
                    "output_tokens": 2, "prompt_tokens": [1, 2, 3, 4]}}


class RunnerInterfaceTests(ControllerHarness):
    TRACE = [("t0", 0, 1), ("t1", 1, 118), ("t2", 2, 312), ("t3", 3, 404), ("t4", 4, 1676)]

    def drive(self, policy_row):
        controller = self.controller(configuration(policy_row))
        coordinator = _FakeCoordinator(self, controller)
        rig = _SpyRig(coordinator.now_us, controller)
        merged = [_trace_row(event_id, index, arrival_s, "gemma-model" if index % 2 else "qwen-model")
                  for event_id, index, arrival_s in self.TRACE]
        state = runner._ArrivalState()
        streams, snapshots = self.root / "streams", self.root / "snapshots"
        streams.mkdir(exist_ok=True)
        snapshots.mkdir(exist_ok=True)
        with mock.patch.object(runner, "CanonicalRuntimeSubmission", lambda **kw: SimpleNamespace(**kw)):
            runner._submit_arrivals(SimpleNamespace(selection_mode="energy-aware"), merged, coordinator, rig, 0,
                                    streams, snapshots, {"qwen-model": "qwen", "gemma-model": "gemma"}, state)
        return controller, coordinator, rig

    def test_online_mode_receives_only_past_arrivals(self):
        controller, coordinator, rig = self.drive(ONLINE)
        arrivals = {event_id: arrival_s * S for event_id, _index, arrival_s in self.TRACE}
        self.assertEqual([call[0] for call in rig.calls], ["arrival_observed"] * len(self.TRACE))
        for kind, (request_id, _model_id, observed_at_us), clock_us in rig.calls:
            self.assertEqual(observed_at_us, clock_us)          # stamped when it happened
            self.assertLessEqual(arrivals[request_id], clock_us)  # the arrival is in the past
        # what the controller knew at every submission: only the arrivals that already happened
        for index, row in enumerate(coordinator.knowledge):
            self.assertIsNone(row["next_arrival_us"])
            self.assertFalse(row["finished"])
            self.assertEqual(len(row["observed"]), index + 1)
            self.assertTrue(all(observed <= row["now_us"] for observed in row["observed"]))
            later = [arrival for arrival in arrivals.values() if arrival > row["now_us"]]
            self.assertFalse(set(later) & set(row["observed"]))
        self.assertEqual(len(coordinator.knowledge[-1]["observed"]), len(self.TRACE))
        self.assertEqual([row["request_id"] for row in controller.telemetry["observed_arrivals"]],
                         [event_id for event_id, _index, _arrival in self.TRACE])
        self.assertIsNone(controller._inputs.next_arrival_us)
        self.assertFalse(controller._inputs.arrivals_finished)

    def test_the_oracle_is_told_every_arrival_before_it_happens(self):
        controller, coordinator, rig = self.drive({**POLICY, "idle": IDLE_NO_EPP})
        calls = [call for call in rig.calls if call[0] == "next_arrival"]
        self.assertEqual(len(calls), len(self.TRACE) + 1)
        future = [call for call in calls[:-1] if call[1] > call[2]]
        self.assertEqual(len(future), len(self.TRACE))  # every one ahead of the clock (first arrival at 1 s)
        self.assertEqual(calls[-1][1], None)             # and told that no arrival follows
        self.assertFalse(any(call[0] == "arrival_observed" for call in rig.calls))
        self.assertTrue(controller._inputs.arrivals_finished)

    def test_runner_helpers_gate_on_the_policy(self):
        class Refusing:
            def __getattr__(self, name):
                raise AssertionError("rig called: " + name)
        for rig in (SimpleNamespace(), SimpleNamespace(configuration=SimpleNamespace(device_power=None))):
            runner._note_next_arrival(rig, 5)
            runner._note_arrival_observed(rig, "r", "m", lambda: 0)
            runner._note_first_token(rig, "r")
        online = configuration(ONLINE)
        refusing = Refusing()
        refusing.__dict__["configuration"] = SimpleNamespace(device_power=online)
        runner._note_next_arrival(refusing, 5)  # online: never forwarded
        runner._note_first_token(refusing, "r")  # no protect_prefill
        legacy = Refusing()
        legacy.__dict__["configuration"] = SimpleNamespace(device_power=configuration(POLICY))
        runner._note_arrival_observed(legacy, "r", "m", lambda: 0)
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("coordinator.wait_for_arrival(request.arrival_us)\n        _note_arrival_observed(rig, "
                      "request.request_id, model_id, coordinator.observed_at_us)\n", source)
        self.assertIn("            _note_first_token(rig, request_id)\n", source)

    def test_result_carries_telemetry_only_with_explicit_arrival_information(self):
        row = {"kind": "DEVICE_POWER_STATE", "device": "desktop-cuda", "from": "OFF", "to": "RESTORED",
               "reason": "capability_probe", "at_us": 0, "command": [], "result": []}
        telemetry = {"schema": "device-power-telemetry-v1", "idle_intervals": [], "execution_waits": []}
        legacy = SimpleNamespace(configuration=SimpleNamespace(device_power=configuration(POLICY)),
                                 device_power_events=(row,), device_power_telemetry=telemetry)
        self.assertEqual(runner._device_power_result(legacy), {"device_power_events": [row]})
        online = SimpleNamespace(configuration=SimpleNamespace(device_power=configuration(ONLINE)),
                                 device_power_events=(row,), device_power_telemetry=telemetry)
        self.assertEqual(runner._device_power_result(online),
                         {"device_power_events": [row], "device_power_telemetry": telemetry})


class TelemetryAnalysisTests(unittest.TestCase):
    PAID_START_NS = 1_000_000_000_000
    PAID_S = 40

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="device-power-online-run-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    @staticmethod
    def event(at_s, source, to, reason, duration_us=120_000):
        return {"kind": "DEVICE_POWER_STATE", "device": "desktop-cuda", "from": source, "to": to, "reason": reason,
                "at_us": int(round(at_s * S)), "command": [["sudo", "-n", SMI, "-rgc"]],
                "result": [{"command": ["sudo", "-n", SMI, "-rgc"], "returncode": 0, "stderr": "", "stdout": "",
                            "readback": None, "duration_us": duration_us}]}

    def write_run(self, directory, *, events, telemetry, power_w=lambda seconds: 20.0):
        directory.mkdir(parents=True, exist_ok=True)
        result = {"paid_start_ns": self.PAID_START_NS, "paid_end_ns": self.PAID_START_NS + self.PAID_S * 10 ** 9,
                  "request_results": [
                      {"replay_arrival_us": 1 * S, "first_token_ns": self.PAID_START_NS + 3 * 10 ** 9,
                       "actual_latency_us": 9 * S},
                      {"replay_arrival_us": 12 * S, "first_token_ns": self.PAID_START_NS + 17 * 10 ** 9,
                       "actual_latency_us": 20 * S}]}
        if events is not None:
            (directory / "DEVICE_POWER_EVENTS.json").write_text(json.dumps(events))
        if telemetry is not None:
            (directory / "DEVICE_POWER_TELEMETRY.json").write_text(json.dumps(telemetry))
        (directory / "RESULT.json").write_text(json.dumps(result))
        rows, energy_uj = [], 5_000_000_000
        for index in range(-2, self.PAID_S * 2 + 3):
            t_ns = self.PAID_START_NS + index * 500_000_000
            energy_uj += 10 * 500_000
            rows.append({"gpu": {"power_mw": int(power_w(index / 2) * 1000), "sample_t_ns": t_ns},
                         "rapl_package": {"energy_uj": energy_uj, "max_energy_range_uj": 262_143_328_850,
                                          "sample_t_ns": t_ns}, "t_ns": t_ns})
        with (directory / "resource-samples.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def online_run(self, directory):
        events = [
            self.event(-1, "OFF", "RESTORED", "capability_probe"),
            self.event(6, "RESTORED", "IDLE_MIN", "idle_timeout"),
            self.event(11.5, "IDLE_MIN", "RESTORED", "arrival_observed"),
            self.event(20.2, "RESTORED", "IDLE_MIN", "predicted_idle"),
            self.event(21.0, "IDLE_MIN", "RESTORED", "late_restore", duration_us=130_000),
            self.event(30.5, "RESTORED", "IDLE_MIN", "idle_timeout"),
            self.event(39.0, "IDLE_MIN", "RESTORED", "end_trace"),
        ]
        telemetry = {
            "schema": "device-power-telemetry-v1", "arrival_information": "online", "horizon_us": 2 * S,
            "idle_intervals": [
                {"start_us": 0, "end_us": 1 * S, "end_reason": "ticket_queued"},          # short, not an opportunity
                {"start_us": 4 * S, "end_us": 12 * S, "end_reason": "execution_active"},  # 8 s, drop after 2 s
                {"start_us": 20 * S, "end_us": 21 * S, "end_reason": "execution_active"},  # false drop
                {"start_us": 24 * S, "end_us": 28 * S, "end_reason": "execution_active"},  # missed entirely
                {"start_us": 28.5 * S, "end_us": 39 * S, "end_reason": "trace_ended"},
            ],
            "execution_waits": [
                {"ticket_id": "a", "request_id": "a", "at_us": 12 * S, "wait_us": 0, "state_before": "RESTORED",
                 "state_after": "RESTORED", "synchronous": False},
                {"ticket_id": "b", "request_id": "b", "at_us": 21 * S, "wait_us": 131_000, "state_before": "IDLE_MIN",
                 "state_after": "RESTORED", "synchronous": True},
            ],
            "observed_arrivals": [{"request_id": "a", "model_id": "m", "observed_at_us": 11_500_000}],
        }
        idle_windows = ((6, 11.5), (20.2, 21.0), (30.5, 39.0))
        power = lambda seconds: 14.0 if any(a <= seconds < b for a, b in idle_windows) else 30.0  # noqa: E731
        self.write_run(directory, events=events, telemetry=telemetry, power_w=power)

    def test_prediction_quality_and_transition_costs(self):
        self.online_run(self.root / "online")
        report = device_power_energy.analyze(self.root / "online")
        quality = report["prediction_quality"]
        self.assertEqual(report["arrival_information"], "online")
        self.assertEqual((quality["idle_intervals"], quality["opportunities"]), (5, 3))
        self.assertEqual(quality["opportunity_us"], 8 * S + 4 * S + 10.5 * S)
        self.assertEqual(quality["captured_us"], 5.5 * S + 0.8 * S + 8.5 * S)
        self.assertEqual(quality["captured_opportunity_us"], 5.5 * S + 8.5 * S)
        self.assertEqual(quality["missed_opportunity_us"], 2.5 * S + 4 * S + 2 * S)
        self.assertEqual(quality["missed_opportunities"], 1)
        self.assertEqual(quality["drop_delay_us"], [2 * S, 2 * S])
        self.assertEqual(quality["idle_min_episodes"], 3)
        self.assertEqual(quality["false_idle_drops"], 1)  # the 0.8 s stay; end_trace does not count
        self.assertEqual(quality["false_idle_drop_exit_reasons"], {"late_restore": 1})
        self.assertEqual((quality["restore_waits"], quality["restore_wait_us"], quality["execution_starts"]), (1, 131_000, 2))
        self.assertEqual(quality["restore_waits_by_request"], [{"request_id": "b", "wait_us": 131_000, "state_before": "IDLE_MIN"}])
        costs = report["transition_costs"]
        self.assertEqual(costs["events"], 6)  # the capability probe is not a transition cost
        self.assertEqual(costs["command_us"], 5 * 120_000 + 130_000)
        self.assertEqual(costs["by_reason"]["late_restore"], {"events": 1, "commands": 1, "command_us": 130_000,
                                                             "max_event_us": 130_000})
        self.assertEqual(report["requests"]["ttft_p50_s"], 2.0)
        self.assertEqual(report["requests"]["ttft_p90_s"], 5.0)
        split = report["idle_split"]
        self.assertAlmostEqual(split["IDLE_MIN:idle"]["seconds"], 14.8, places=6)
        self.assertLess(split["IDLE_MIN:idle"]["gpu_w"], 16.0)  # 14 W samples, edges interpolated
        self.assertAlmostEqual(split["RESTORED:idle"]["seconds"], 1 + 2 + 0.5 + 0.2 + 4 + 2, places=6)
        self.assertGreater(split["RESTORED:busy"]["gpu_w"], 25.0)
        self.assertEqual(sum(row["seconds"] for row in split.values()), 40.0)
        text = device_power_energy.render(report)
        self.assertIn("prediction quality (online, horizon 2.0 s)", text)
        self.assertIn("late_restore", text)

    def test_compare_three_arms(self):
        self.online_run(self.root / "online")
        self.write_run(self.root / "always-on", events=None, telemetry=None, power_w=lambda seconds: 30.0)
        report = device_power_energy.compare([("always-on", self.root / "always-on"), ("online", self.root / "online")])
        always, online = report["runs"]
        self.assertEqual(always["state_seconds"], {"UNCONTROLLED": 40.0})
        self.assertIsNone(always["captured_idle_s"])
        self.assertAlmostEqual(always["gpu_kj"], 1.2, delta=0.02)
        self.assertLess(online["gpu_kj"], always["gpu_kj"])
        self.assertEqual(online["false_idle_drops"], 1)
        self.assertAlmostEqual(online["restore_wait_s"], 0.131)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(device_power_energy.main(["--compare", "always-on=" + str(self.root / "always-on"),
                                                       "online=" + str(self.root / "online")]), 0)
        self.assertIn("always-on", printed.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            device_power_energy.main(["--compare", "online=" + str(self.root / "online"), "--json"])
        self.assertEqual(json.loads(printed.getvalue())["schema"], "s42-device-power-compare-v1")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            device_power_energy.main(["--compare", "nolabel"])


class RigPassThroughTests(unittest.TestCase):
    def test_rig_forwards_observations_and_telemetry_only_with_a_controller(self):
        from research_dev.scheduler.adapters.heterogeneous_rig import HeterogeneousPhysicalRig
        rig = HeterogeneousPhysicalRig.__new__(HeterogeneousPhysicalRig)  # tests build rigs without __init__
        rig.note_arrival_observed("r", "m", 0)
        rig.note_first_token("r")
        self.assertIsNone(rig.device_power_telemetry)
        calls = []
        rig._device_power = SimpleNamespace(
            note_arrival_observed=lambda *args: calls.append(("arrival", args)),
            note_first_token=lambda *args: calls.append(("first", args)),
            telemetry={"schema": "x"})
        rig.note_arrival_observed("r", "m", 5)
        rig.note_first_token("r")
        self.assertEqual(calls, [("arrival", ("r", "m", 5)), ("first", ("r",))])
        self.assertEqual(rig.device_power_telemetry, {"schema": "x"})


def _recorded_result():
    """Three requests: 000 loads then decodes; 001 arrives during it and joins; 002 arrives after a
    120 s idle gap (paid window 400 s, clock in us, paid_start_ns = 5e12)."""
    start_ns = 5 * 10 ** 12

    def row(request_id, arrival_s, acquired_s, started_s, first_s, finished_s):
        return {"request_id": request_id, "model_id": "qwen", "replay_arrival_us": int(arrival_s * S),
                "scheduling_overhead": {"snapshot_capture_ns": 5_000_000, "scheduler_submit_ns": 995_000_000},
                "dispatch_receipts": [{"status": "ACQUIRED", "observed_at_us": int(acquired_s * S)}],
                "first_token_ns": start_ns + int(first_s * 10 ** 9),
                "execution_command": {"participants": [{"device_id": "desktop-cpu"}, {"device_id": "desktop-cuda"}]},
                "completion": {"execution_receipt": {"started_us": int(started_s * S), "finished_us": int(finished_s * S)}}}
    return {"paid_start_ns": start_ns, "paid_end_ns": start_ns + 400 * 10 ** 9, "request_results": [
        row("000", 1, 3, 43, 50, 150), row("001", 60, 61, 61.5, 70, 170), row("002", 290, 291, 291.5, 295, 390)]}


class ReplayToolTests(unittest.TestCase):
    ONLINE = {**ONLINE, "decode_cap": {"sm_max_mhz": 1200, "protect_prefill": True}, "load_min": True}
    ORACLE = {**POLICY, "idle": IDLE_NO_EPP, "arrival_information": "oracle",
              "decode_cap": {"sm_max_mhz": 1200, "protect_prefill": True}, "load_min": True}

    def test_timeline_extraction(self):
        timeline = device_power_replay.extract_timeline(_recorded_result())
        self.assertEqual(timeline["end_us"], 400 * S)
        first = timeline["requests"][0]
        self.assertEqual((first["submitted_us"], first["acquired_us"], first["started_us"], first["first_token_us"]),
                         (2 * S, 3 * S, 43 * S, 50 * S))
        self.assertEqual([row["load"] for row in timeline["requests"]], [True, False, False])
        activity = device_power_replay.activity_timeline(timeline)
        self.assertEqual(activity[:4], [(0, 3 * S, "idle"), (3 * S, 43 * S, "load"), (43 * S, 50 * S, "prefill"),
                                        (50 * S, 61_500_000, "decode")])
        self.assertIn((170 * S, 291_500_000, "idle"), activity)

    def test_variants_on_one_timeline(self):
        report = device_power_replay.replay(_recorded_result(), [
            ("always-on", None), ("oracle", configuration(self.ORACLE)), ("online", configuration(self.ONLINE)),
            ("capped", configuration({**POLICY, "idle": IDLE_NO_EPP, "decode_cap": {"sm_max_mhz": 1200}}))])
        always, oracle, online, capped = report["variants"]
        self.assertEqual(always["activity_state_seconds"]["idle"], {"UNCONTROLLED": report["activity_seconds"]["idle"]})
        self.assertIsNone(always["prediction_quality"])
        self.assertEqual(always["gpu_kj_saved_vs_uncontrolled"], 0)
        # the 121.5 s gap (170 -> 291.5 s): the oracle drops at once, online once the GPU has been idle
        # for the 2 s fallback (one gap of 59 s observed: no prediction), and both restore before 002
        # arrives. The oracle, told at 002's submission that no arrival follows, drops again in the
        # 0.5 s ACQUIRED -> start hand-off and pays a late restore; the online settle time avoids it.
        for row, drop_delay, false_drops in ((oracle, 0, 1), (online, 2 * S, 0)):
            quality = row["prediction_quality"]
            self.assertEqual(quality["false_idle_drops"], false_drops)
            self.assertIn(drop_delay, quality["drop_delay_us"])
            self.assertGreater(row["gpu_kj_saved_vs_uncontrolled"], 0)
            self.assertEqual(row["capped_prefill_s"], 0.0)
            self.assertEqual(row["activity_state_seconds"]["load"], {"LOAD_MIN": 40.0})
        self.assertEqual(oracle["prediction_quality"]["false_idle_drop_exit_reasons"], {"late_restore": 1})
        self.assertEqual(online["synchronous_restores"], 1)  # only the joiner's prefill restore
        self.assertIn("arrival_observed", online["reasons"])
        self.assertIn("predictive_restore", oracle["reasons"])
        self.assertGreaterEqual(oracle["prediction_quality"]["captured_us"], online["prediction_quality"]["captured_us"])
        # without protect_prefill the cap covers the prompts (and the joiner's wait for its first token)
        self.assertAlmostEqual(capped["capped_prefill_s"], 7 + 8.5 + 3.5, places=6)
        self.assertEqual(online["synchronous_restores"], online["reasons"].get("late_restore", 0)
                         + online["reasons"].get("prefill_restore", 0))
        self.assertIn("prefill_restore", online["reasons"])  # 001 joins 000 while it is capped
        text = device_power_replay.render(report)
        self.assertIn("online", text)

    def test_cli_reads_campaigns_and_policy_files(self):
        root = Path(tempfile.mkdtemp(prefix="device-power-replay-"))
        self.addCleanup(shutil.rmtree, root, True)
        (root / "RESULT.json").write_text(json.dumps(_recorded_result()))
        (root / "campaign.json").write_text(json.dumps({"schema": "research-scheduler-campaign-v1",
                                                        "device_power": self.ONLINE}))
        (root / "plain-campaign.json").write_text(json.dumps({"schema": "research-scheduler-campaign-v1"}))
        (root / "policy.json").write_text(json.dumps(self.ORACLE))
        self.assertIsNone(device_power_replay.load_variant(str(root / "plain-campaign.json")))
        self.assertIsNone(device_power_replay.load_variant("none"))
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(device_power_replay.main([
                "--result", str(root / "RESULT.json"), "--variant", "always-on=none",
                "--variant", "online=" + str(root / "campaign.json"), "--variant", "oracle=" + str(root / "policy.json"),
                "--json"]), 0)
        report = json.loads(printed.getvalue())
        self.assertEqual(report["schema"], "s42-device-power-replay-v1")
        self.assertEqual([row["label"] for row in report["variants"]], ["always-on", "online", "oracle"])
        self.assertEqual(report["variants"][1]["policy"]["arrival_information"], "online")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            device_power_replay.main(["--result", str(root / "RESULT.json"), "--variant", "broken"])


if __name__ == "__main__":
    unittest.main()
