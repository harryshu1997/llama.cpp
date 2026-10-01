#!/usr/bin/env python3
"""Joint planner shadow mode: configuration, journal-stream mirror, planning off the scheduler's
locks, error isolation, runner wiring, and byte-identical scheduler decisions with the shadow on."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from research_dev.scheduler._internal.joint_planner_model import measured_eval_v2_cost_model
from research_dev.scheduler._internal.joint_planner_shadow import (
    JointPlannerShadow,
    JointPlannerShadowConfig,
    JointPlannerShadowError,
    ShadowEpoch,
    default_model_key,
    shadow_config_from_policy_json,
    simulator_from_epoch,
)
from research_dev.scheduler._internal.joint_planner_sim import SimRequest
from research_dev.scheduler.campaigns.burstgpt import runner
from research_dev.scheduler.configuration.campaign import _dispatch_policy
from research_dev.scheduler.configuration.common import SchedulerConfigurationError

import test_dispatch_policy as dispatch_harness

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "campaigns" / "burstgpt"))
import joint_planner_eval as evaluation  # noqa: E402


QWEN = "qwen3-14b-q4km-dequant-f16"
GEMMA = "gemma-4-12b-q40-dequant-f16"


def ticket(request_id, model_id, arrival_s, output_tokens=120, *, load=False, start_s=None):
    transitions = ()
    if load:
        transitions = (SimpleNamespace(transition_id="load:" + model_id + ":physical:hot:desktop:cold",
                                       device_id="desktop-cuda"),)
    return SimpleNamespace(
        request=SimpleNamespace(request_id=request_id, arrival_us=int(arrival_s * 1e6), input_tokens=100,
                                output_tokens=output_tokens),
        model=SimpleNamespace(model_id=model_id, artifact_sha256="sha256:" + model_id),
        decision=SimpleNamespace(start_us=int((start_s or arrival_s) * 1e6), finish_upper_us=0),
        execution_plan=SimpleNamespace(route_id="route:" + model_id, transitions=transitions),
        dispatch_state="QUEUED",
    )


def placement(progress=None, sessions=None):
    controller = SimpleNamespace(_request_decode_progress=progress or {}, _phone_session_states=sessions or {})
    return SimpleNamespace(_model_placement_controller=controller)


class ConfigurationTests(unittest.TestCase):
    def test_shadow_is_the_only_mode(self) -> None:
        self.assertEqual(JointPlannerShadowConfig.from_json({"mode": "shadow"}).budget_ms, 50.0)
        for value in ({"mode": "active"}, {}, {"mode": "shadow", "budget_ms": 0},
                      {"mode": "shadow", "depth": -1}, {"mode": "shadow", "objective": "latency"},
                      {"mode": "shadow", "extra": 1}, ["shadow"]):
            with self.subTest(value=value), self.assertRaises(JointPlannerShadowError):
                JointPlannerShadowConfig.from_json(value)

    def test_campaign_accepts_the_key_and_keeps_given_fields(self) -> None:
        checked = _dispatch_policy({"work_conserving_admission": True,
                                    "joint_planner": {"mode": "shadow", "budget_ms": 20}})
        self.assertEqual(checked["joint_planner"], {"budget_ms": 20, "mode": "shadow"})
        self.assertEqual(json.loads(json.dumps(dict(checked)))["joint_planner"]["mode"], "shadow")
        with self.assertRaises(SchedulerConfigurationError):
            _dispatch_policy({"joint_planner": {"mode": "active", "extra": 1}})
        with self.assertRaises(SchedulerConfigurationError):
            _dispatch_policy({"joint_planner": {"mode": "replay"}})
        self.assertNotIn("joint_planner", _dispatch_policy({"work_conserving_admission": True}))

    def test_runner_splits_the_shadow_from_the_dispatch_policy(self) -> None:
        text = json.dumps({"work_conserving_admission": True, "joint_planner": {"mode": "shadow"}})
        policy = runner._dispatch_policy_from_json(text)
        self.assertTrue(policy.work_conserving_admission)
        self.assertIsNone(runner._dispatch_policy_from_json(json.dumps({"joint_planner": {"mode": "shadow"}})))
        self.assertIsInstance(runner._joint_planner_shadow_from_json(text), JointPlannerShadow)
        self.assertIsNone(runner._joint_planner_shadow_from_json(json.dumps({"model_affinity": False})))
        self.assertIsNone(runner._joint_planner_shadow_from_json(None))
        rest, config = shadow_config_from_policy_json({"model_affinity": False})
        self.assertEqual((rest, config), ({"model_affinity": False}, None))
        scheduler = SimpleNamespace(joint_planner_shadow=lambda: None)
        self.assertEqual(runner._joint_planner_shadow_result(scheduler), {})

    def test_model_keys(self) -> None:
        self.assertEqual(default_model_key(QWEN), "qwen")
        self.assertEqual(default_model_key(GEMMA), "gemma")
        self.assertEqual(default_model_key("llama-3.2-1b-instruct-q4_0"), "llama")
        self.assertIsNone(default_model_key("mistral-7b"))


class MirrorTests(unittest.TestCase):
    def shadow(self) -> JointPlannerShadow:
        return JointPlannerShadow(JointPlannerShadowConfig(mode="shadow", budget_ms=200.0), synchronous=True)

    def test_journal_stream_builds_the_planner_state(self) -> None:
        shadow = self.shadow()
        a = ticket("001", QWEN, 0.0, 124, load=True)
        b = ticket("003", QWEN, 90.0, 262)
        shadow.observe_ticket(placement(), "DECISION", a, 0, "QUEUED")
        shadow.observe_ticket(placement(), "ACQUIRED", a, 1_000_000, "ACQUIRED")
        sessions = {"HTP0": SimpleNamespace(endpoint="session://op15-phone/HTP0", state="READY",
                                            resident_artifact_sha256="sha256:" + QWEN,
                                            active_helper_references=("helper",))}
        shadow.observe_ticket(placement({"001": (124, 40)}, sessions), "DECISION", b, 90_000_000, "QUEUED")
        shadow.observe_ticket(placement(), "COMPLETED", a, 150_000_000, "COMPLETED")
        records = shadow.records()
        self.assertEqual([r["event_kind"] for r in records], ["DECISION", "ACQUIRED", "DECISION", "COMPLETED"])
        self.assertTrue(all("error" not in r for r in records))
        self.assertEqual(records[0]["sequential_emulated"], ["switch qwen", "provision op15 qwen"])
        self.assertEqual(records[1]["state"]["loading"], "qwen")
        joined = records[2]
        self.assertEqual(joined["state"]["queued"], ["003"])
        self.assertEqual(joined["state"]["phone_model"], "qwen")
        self.assertTrue(joined["state"]["assisted"])
        self.assertEqual(joined["live"]["route_id"], "route:" + QWEN)
        summary = shadow.summary()
        self.assertEqual(summary["epochs"], 4)
        self.assertEqual(summary["errors"], 0)
        artifact = shadow.artifact()
        self.assertEqual(len(artifact["records"]), 4)
        json.dumps(artifact)

    def test_capture_errors_are_recorded_not_raised(self) -> None:
        shadow = self.shadow()
        shadow.observe_ticket(placement(), "DECISION", SimpleNamespace(request=None), 5, "QUEUED")
        self.assertEqual(shadow.summary()["errors"], 1)
        self.assertEqual(shadow.records()[0]["error"], "CAPTURE_FAILED")

    def test_unmodelled_models_are_listed_and_skipped(self) -> None:
        shadow = self.shadow()
        shadow.observe_ticket(placement(), "DECISION", ticket("m", "mistral-7b", 0.0), 0, "QUEUED")
        self.assertEqual(shadow.records()[0]["state"]["queued"], [])

    def test_worker_thread_drains_on_close(self) -> None:
        shadow = JointPlannerShadow(JointPlannerShadowConfig(mode="shadow"))
        for index in range(5):
            shadow.observe_ticket(placement(), "DECISION", ticket("r%d" % index, GEMMA, index), index * 10**6, "QUEUED")
        shadow.close()
        self.assertEqual(shadow.summary()["epochs"], 5)
        shadow.observe_ticket(placement(), "DECISION", ticket("late", GEMMA, 9.0), 9 * 10**6, "QUEUED")
        self.assertEqual(shadow.summary()["epochs"], 5)

    def test_simulator_from_epoch(self) -> None:
        cost = measured_eval_v2_cost_model()
        epoch = ShadowEpoch(
            sequence=1, event_kind="DECISION", lifecycle_state="QUEUED", now_s=100.0, request_id="b",
            live={}, queue=(SimRequest("b", "qwen", 99.0, 50, 200),),
            rows=((SimRequest("a", "qwen", 10.0, 50, 300), 150.0),), resident="qwen", loading=None,
            load_end_s=0.0, assisted=True, phone_model="qwen", phone_loading=None, phone_ready_s=0.0,
        )
        sim = simulator_from_epoch(epoch, cost)
        self.assertEqual(sim.server.resident, "qwen")
        self.assertTrue(sim.server.assisted)
        self.assertEqual([r.request_id for r in sim.queue], ["b"])
        self.assertEqual(sim.server.rows[0].tokens_left, 150.0)
        loading = simulator_from_epoch(
            ShadowEpoch(**{**epoch.__dict__, "rows": (), "loading": "gemma", "load_end_s": 130.0,
                           "phone_model": None, "phone_loading": "gemma", "phone_ready_s": 120.0}), cost)
        self.assertEqual((loading.server.loading, loading.primary.loading), ("gemma", "gemma"))


class LiveSchedulerTests(unittest.TestCase):
    """The hook reads the real scheduler's journal stream and changes no decision."""

    def run_scheduler(self, shadow):
        case = dispatch_harness.DispatchPolicySchedulerTests("test_same_model_arrival_joins_the_running_server")
        case.setUp()
        self.addCleanup(case.directory.cleanup)
        scheduler, model_a, model_b, hot = case.scheduler(dispatch_harness.WORK_CONSERVING)
        if shadow is not None:
            scheduler.configure_joint_planner_shadow(shadow)
        tickets = [
            case.submit(scheduler, model_a, hot, "a1", 1_000, 640),
            case.submit(scheduler, model_b, hot, "b1", 1_100, 8),
            case.submit(scheduler, model_a, hot, "a2", 1_200, 4),
        ]
        return scheduler, tickets

    def test_decisions_are_byte_identical_with_the_shadow(self) -> None:
        plain, plain_tickets = self.run_scheduler(None)
        keys = {"model-a": "qwen", "model-b": "gemma"}
        shadow = JointPlannerShadow(JointPlannerShadowConfig(mode="shadow"), model_key=keys.get,
                                    synchronous=True)
        observed, observed_tickets = self.run_scheduler(shadow)
        self.assertIsNone(plain.joint_planner_shadow())
        self.assertIs(observed.joint_planner_shadow(), shadow)
        self.assertEqual(plain.runtime_decision_log_bytes(), observed.runtime_decision_log_bytes())
        self.assertEqual([t.decision.start_us for t in plain_tickets], [t.decision.start_us for t in observed_tickets])
        records = shadow.records()
        self.assertGreaterEqual(len(records), 3)
        self.assertTrue(all("error" not in r for r in records), records)
        self.assertEqual({r["request_id"] for r in records}, {"a1", "b1", "a2"})


class ShadowReplayTests(unittest.TestCase):
    def test_recorded_s2a_stream_through_the_hook(self) -> None:
        rows, summary = evaluation.shadow_replay_rows(evaluation.load_runs()["s2a"], measured_eval_v2_cost_model())
        self.assertEqual((summary["epochs"], summary["errors"]), (42, 0))
        self.assertIn(["404.0", "DECISION", "003", "park 003", "admit 003"], [row[:5] for row in rows])
        self.assertIn(["458.0", "DECISION", "004", "park 004", "admit 004"], [row[:5] for row in rows])


if __name__ == "__main__":
    unittest.main()
