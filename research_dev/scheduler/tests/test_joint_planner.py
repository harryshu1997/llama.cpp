#!/usr/bin/env python3
"""Joint planner: cost model, discrete-event simulator, sequential emulation, planner, exact search,
and the offline evaluator's calibration against the recorded longtail_eval_v2 runs."""

from __future__ import annotations

from dataclasses import replace
import math
import sys
from pathlib import Path
import unittest
from unittest import mock

from research_dev.scheduler._internal import joint_planner

from research_dev.scheduler._internal.joint_planner import (
    JointPlanner,
    JointPlannerConfig,
    JointPlannerError,
    Phase,
    SchedulePolicy,
    clairvoyant_leaf,
    exhaustive_optimum,
    joint_candidates,
    ordered_phase_schedules,
    schedule_space_optimum,
)
from research_dev.scheduler._internal.joint_planner_model import (
    JointPlannerModelError,
    PlannerConstraints,
    measured_eval_v2_cost_model,
)
from research_dev.scheduler._internal.joint_planner_sim import (
    Admit,
    LegacyPolicy,
    Park,
    Provision,
    SequentialPolicy,
    SetAssist,
    SimOptions,
    SimRequest,
    SimulationError,
    Simulator,
    Switch,
    percentile,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "campaigns" / "burstgpt"))
import joint_planner_eval as evaluation  # noqa: E402


COST = measured_eval_v2_cost_model()


def request(rid, model, arrival, tokens=100, prompt=100):
    return SimRequest(rid, model, arrival, prompt, tokens)


class Idle:
    def decide(self, sim):
        return []


class CostModelTests(unittest.TestCase):
    def test_tables_are_validated(self) -> None:
        costs = COST.model("qwen")
        with self.assertRaises(JointPlannerModelError):
            replace(costs, desktop_step_s=(0.6,))
        with self.assertRaises(JointPlannerModelError):
            replace(costs, helper_devices=())
        with self.assertRaises(JointPlannerModelError):
            PlannerConstraints(maximum_latency_ppm=900_000)
        self.assertEqual(costs.step_s(9, True), costs.assisted_step_s[-1])
        self.assertAlmostEqual(costs.load_energy_j(), 1021.0 + 13.3 * 74.2)

    def test_batching_lowers_energy_per_token(self) -> None:
        costs = COST.model("qwen")
        per_token = [costs.step_s(b, True) * costs.power_w(b, True) / b for b in range(1, 5)]
        self.assertEqual(per_token, sorted(per_token, reverse=True))
        self.assertLess(COST.best_token_energy_j("qwen"), per_token[0])
        self.assertIn("qwen", COST.to_json()["models"])


class SimulatorTests(unittest.TestCase):
    def test_idle_energy_is_exact(self) -> None:
        sim = Simulator([request("r", "gemma", 10.0)], COST, SimOptions(phones_enabled=False, horizon_s=10.0))
        result = sim.run(Idle())
        self.assertAlmostEqual(result.host_j, 10.0 * COST.idle_unloaded_w)
        self.assertEqual(result.unfinished, ("r",))

    def test_single_request_desktop_energy(self) -> None:
        costs = COST.model("gemma")
        result = Simulator([request("r", "gemma", 0.0, tokens=50, prompt=20)], COST,
                           SimOptions(phones_enabled=False)).run(LegacyPolicy())
        completion = result.completions["r"]
        expected_end = costs.load_s + costs.prefill_s(20) + 50 * costs.step_s(1, False)
        self.assertAlmostEqual(completion.end_s, expected_end, places=6)
        expected = (costs.load_energy_j() + costs.prefill_s(20) * costs.power_w(1, False)
                    + 50 * costs.step_s(1, False) * costs.power_w(1, False))
        self.assertAlmostEqual(result.host_j, expected, places=3)
        self.assertEqual(completion.assisted_tokens, 0.0)

    def test_join_batches_and_park_serializes(self) -> None:
        requests = [request("a", "qwen", 0.0, tokens=200), request("b", "qwen", 1.0, tokens=200)]
        for action, expected_rows in ((Admit, 2), (Park, 1)):
            sim = Simulator(requests, COST, SimOptions(phones_enabled=False))
            sim._process_events()
            sim.apply(Switch("qwen"))
            while sim.server.loading or len(sim.queue) < 2:
                sim.advance()
            sim.apply(Admit(("a",)))
            sim.apply(action(("b",)))
            self.assertEqual(len(sim.server.decoding()), expected_rows)
            result = sim.run(SequentialPolicy(helpers=False))
            if action is Admit:
                self.assertLess(result.completions["b"].end_s, 200 * 0.626 * 1.5 + 100)
            else:
                self.assertGreaterEqual(result.completions["b"].decode_start_s, result.completions["a"].end_s - 1e-6)

    def test_invalid_actions_raise(self) -> None:
        sim = Simulator([request("a", "qwen", 0.0)], COST)
        sim._process_events()
        with self.assertRaises(SimulationError):
            sim.apply(Admit(("a",)))
        sim.apply(Switch("qwen"))
        with self.assertRaises(SimulationError):
            sim.apply(Switch("gemma"))
        with self.assertRaises(SimulationError):
            sim.apply(Provision("pixel", "qwen"))
        with self.assertRaises(SimulationError):
            sim.apply(SetAssist(True))

    def test_cohort_lease_lag_defers_and_fix_retries(self) -> None:
        requests = [request("g1", "gemma", 0.0, 60), request("g2", "gemma", 0.5, 60),
                    request("q1", "qwen", 5.0, 200)]
        bug = Simulator(requests, COST).run(SequentialPolicy(retry_reprovision_on_release=False))
        fixed = Simulator(requests, COST).run(SequentialPolicy())
        self.assertIn((bug.completions["g2"].end_s, "defer op15 qwen"),
                      [(t, d) for t, d in bug.decisions if d.startswith("defer")] or [(None, None)])
        self.assertEqual(bug.completions["q1"].assisted_tokens, 0.0)
        self.assertAlmostEqual(fixed.completions["q1"].assisted_tokens, 200.0, places=6)
        self.assertLess(fixed.host_j, bug.host_j)

    def test_late_adoption_and_thermal_exclusion(self) -> None:
        requests = [request("q1", "qwen", 0.0, 400)]
        slow_phone = replace(COST, phones={**COST.phones, "op15": replace(COST.phones["op15"], session_load_s=40.0)})
        adopted = Simulator(requests, slow_phone).run(SequentialPolicy())
        self.assertGreater(adopted.completions["q1"].assisted_tokens, 0.0)
        self.assertLess(adopted.completions["q1"].assisted_tokens, 400.0)
        excluded = Simulator(requests, COST, SimOptions(thermal_exclusions=(("op15", 0.0, 5000.0),))).run(
            SequentialPolicy())
        self.assertEqual(excluded.completions["q1"].assisted_tokens, 0.0)

    def test_recorded_load_durations_are_used_per_model(self) -> None:
        options = SimOptions(phones_enabled=False, load_s_by_model=(("gemma", (5.0,)),))
        result = Simulator([request("g", "gemma", 0.0, 10)], COST, options).run(LegacyPolicy())
        self.assertLess(result.completions["g"].end_s, 5.0 + COST.model("gemma").prefill_s(100) + 10)

    def test_percentile(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertTrue(math.isnan(percentile([], 50)))


class SequentialEmulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests = evaluation.trace_requests(evaluation.load_runs()["s2a"])

    def test_reproduces_the_s2a_structure(self) -> None:
        result = Simulator(self.requests, COST, evaluation.s2a_options()).run(SequentialPolicy())
        decisions = [d for _, d in result.decisions]
        self.assertIn("park 003", decisions)
        self.assertIn("park 004", decisions)
        self.assertIn("admit 002,006 assisted", decisions)
        self.assertIn("defer op15 qwen", decisions)
        self.assertIn("admit 005,007 desktop", decisions)
        self.assertIn("admit 011,012 assisted", decisions)
        self.assertEqual(result.loads, 7)
        self.assertEqual(result.completions["005"].assisted_tokens, 0.0)

    def test_legacy_runs_the_loader_alone(self) -> None:
        result = Simulator(self.requests, COST, SimOptions(phones_enabled=False)).run(LegacyPolicy())
        decisions = [d for _, d in result.decisions]
        self.assertIn("admit 003 desktop", decisions)
        self.assertIn("admit 004,005 desktop", decisions)
        self.assertEqual(result.loads, 9)


class PlannerTests(unittest.TestCase):
    def scenario(self, name):
        requests = evaluation.trace_requests(evaluation.load_runs()["s2a"])
        return next(s for s in evaluation.scenarios(requests, COST) if s.name.startswith(name))

    def test_config_validation(self) -> None:
        with self.assertRaises(JointPlannerError):
            JointPlannerConfig(objective="latency")
        with self.assertRaises(JointPlannerError):
            JointPlannerConfig.from_json({"budget_ms": 10, "unknown": 1})
        self.assertEqual(JointPlannerConfig.from_json({"mode": "shadow", "depth": 1}).depth, 1)

    def test_candidate_zero_is_the_sequential_decision(self) -> None:
        sim = Simulator([request("a", "qwen", 0.0)], COST)
        sim._process_events()
        candidates = joint_candidates(sim.clone(keep_future=False), SequentialPolicy())
        self.assertEqual(candidates[0], (Switch("qwen"), Provision("op15", "qwen")))
        self.assertIn((Switch("qwen"),), candidates)

    def test_joins_after_the_cohort_window(self) -> None:
        scenario = self.scenario("F1 ")
        planner = JointPlanner(JointPlannerConfig())
        planned = Simulator(list(scenario.requests), COST).run(planner)
        sequential = Simulator(list(scenario.requests), COST).run(SequentialPolicy())
        self.assertIn("park 003", [d for _, d in sequential.decisions])
        deviations = [p for _, p in planner.plans if p.deviates]
        self.assertTrue(any(Admit(("003",)) in p.actions for p in deviations))
        self.assertLess(planned.fleet_j, sequential.fleet_j)
        self.assertLess(planned.completions["003"].end_s, sequential.completions["003"].end_s)

    def test_recovers_a_missing_release_retry(self) -> None:
        scenario = self.scenario("F2c ")
        bug = Simulator(list(scenario.requests), COST).run(scenario.sequential)
        planned = Simulator(list(scenario.requests), COST).run(JointPlanner(JointPlannerConfig(), scenario.sequential))
        self.assertEqual(bug.completions["005"].assisted_tokens, 0.0)
        self.assertGreater(planned.completions["005"].assisted_tokens, 0.0)
        self.assertLess(planned.fleet_j, bug.fleet_j)

    def test_thermal_window_blocks_provisioning_until_it_clears(self) -> None:
        for name, assisted in (("F2 ", False), ("F2b ", True)):
            scenario = self.scenario(name)
            for policy in (scenario.sequential, JointPlanner(JointPlannerConfig(), scenario.sequential)):
                result = Simulator(list(scenario.requests), COST, scenario.options).run(policy)
                with self.subTest(case=name, policy=type(policy).__name__):
                    self.assertEqual(result.completions["005"].assisted_tokens > 0, assisted)
        sim = Simulator([request("q", "qwen", 0.0)], COST, SimOptions(thermal_exclusions=(("op15", 0.0, 9.0),)))
        sim._process_events()
        with self.assertRaises(SimulationError):
            sim.apply(Provision("op15", "qwen"))

    def test_never_worse_than_sequential_on_the_trace(self) -> None:
        requests = evaluation.trace_requests(evaluation.load_runs()["s2a"])
        sequential = Simulator(requests, COST).run(SequentialPolicy())
        planner = JointPlanner(JointPlannerConfig(budget_ms=100.0))
        planned = Simulator(requests, COST).run(planner)
        self.assertLessEqual(planned.fleet_j, sequential.fleet_j + 1.0)
        latencies = sorted(c.latency_s for c in planned.completions.values())
        self.assertLessEqual(percentile(latencies, 90),
                             percentile([c.latency_s for c in sequential.completions.values()], 90) + 1.0)
        self.assertTrue(all(p.fallback_reason is None for _, p in planner.plans))

    def test_epoch_delay_bound_blocks_idle_holds(self) -> None:
        sim = Simulator([request("g", "gemma", 1.0, 300)], COST)
        sim._process_events()
        sim.advance()
        strict = JointPlanner(JointPlannerConfig()).plan(sim)
        self.assertFalse(strict.deviates)
        loose = JointPlanner(JointPlannerConfig(constraints=PlannerConstraints(maximum_epoch_delay_s=1e6))).plan(sim)
        self.assertTrue(loose.deviates)
        self.assertEqual(loose.actions, (Provision("op15", "gemma"),))

    def test_budget_exhaustion_keeps_the_incumbent(self) -> None:
        scenario = self.scenario("W1 ")
        sim = Simulator(list(scenario.requests), COST)
        sim._process_events()
        sim.advance()
        plan = JointPlanner(JointPlannerConfig(budget_ms=1e-6)).plan(sim)
        self.assertTrue(plan.budget_exhausted)
        self.assertEqual(plan.actions, plan.incumbent)

    def test_search_failure_falls_back_to_the_cascade(self) -> None:
        sim = Simulator([request("a", "qwen", 0.0), request("b", "gemma", 0.0)], COST)
        sim._process_events()
        planner = JointPlanner(JointPlannerConfig())
        with mock.patch.object(joint_planner, "references_for", side_effect=RuntimeError("boom")):
            plan = planner.plan(sim)
        self.assertEqual(plan.fallback_reason, "SEARCH_FAILED: RuntimeError: boom")
        self.assertEqual(plan.actions, plan.incumbent)
        self.assertEqual(plan.actions, joint_planner.epoch_decision(sim, SequentialPolicy()))
        with mock.patch.object(joint_planner, "joint_candidates", side_effect=ValueError("bad")):
            plan = planner.plan(sim)
        self.assertTrue(plan.fallback_reason.startswith("CANDIDATES_FAILED"))


class ExactSearchTests(unittest.TestCase):
    def test_schedule_space_counts(self) -> None:
        requests = [request("a", "qwen", 0.0), request("b", "qwen", 1.0), request("g", "gemma", 2.0)]
        schedules = ordered_phase_schedules(requests, COST, phones=False)
        self.assertEqual(len(schedules), 8)
        with_helpers = ordered_phase_schedules(requests, COST)
        self.assertEqual(len(with_helpers), 2 * 2 * 2 + 6 * 2 * 2 * 2)

    def test_schedule_policy_holds_for_a_future_member(self) -> None:
        requests = [request("a", "qwen", 0.0, 50), request("b", "qwen", 200.0, 50)]
        policy = SchedulePolicy((Phase("qwen", ("a", "b"), False),))
        result = Simulator(requests, COST, SimOptions(phones_enabled=False)).run(policy)
        self.assertEqual(result.loads, 1)
        self.assertEqual(set(result.completions), {"a", "b"})

    def test_optimum_is_no_worse_than_the_policies(self) -> None:
        requests = evaluation.trace_requests(evaluation.load_runs()["s2a"])
        scenario = next(s for s in evaluation.scenarios(requests, COST) if s.name.startswith("W4"))
        reqs = list(scenario.requests)
        enumerated = schedule_space_optimum(reqs, COST)
        bnb = exhaustive_optimum(reqs, COST, node_limit=5_000)
        self.assertTrue(bnb.complete)
        for policy in (SequentialPolicy(), JointPlanner(JointPlannerConfig())):
            sim = Simulator(reqs, COST)
            sim.run(policy)
            leaf = clairvoyant_leaf(reqs, COST, SimOptions(), sim)
            self.assertLessEqual(min(enumerated.score, bnb.score), leaf.score + 1e-6)
        self.assertTrue(enumerated.leaf.feasible)


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runs = evaluation.load_runs()
        self.totals = evaluation.load_totals()

    def test_fixture_covers_the_recorded_runs(self) -> None:
        self.assertEqual(sorted(self.runs), ["s1a", "s1b", "s1c", "s1d", "s2a"])
        self.assertEqual(len(self.runs["s2a"]["requests"]), 14)
        self.assertEqual(sum(q["output_tokens"] for q in self.runs["s2a"]["requests"]), 3604)
        self.assertEqual(len(self.totals["legacy"]["requests"]), 14)

    def test_derived_parameters_match_the_frozen_model(self) -> None:
        derived = evaluation.derive_parameters(self.runs)
        for model in ("gemma", "qwen"):
            seconds, fixed, slope, _ = derived["loads"][model]
            costs = COST.model(model)
            self.assertAlmostEqual(seconds, costs.load_s, delta=0.1)
            self.assertAlmostEqual(fixed, costs.load_fixed_j, delta=1.0)
            self.assertAlmostEqual(slope, costs.load_power_w, delta=0.1)
            intercept, per_token, _ = derived["prefill"][model]
            self.assertAlmostEqual(intercept, costs.prefill_fixed_s, delta=0.01)
            self.assertAlmostEqual(per_token, costs.prefill_s_per_token, delta=1e-5)
        period, _ = derived["periods"]["gemma-assisted-b1"]
        self.assertAlmostEqual(period, COST.model("gemma").step_s(1, True), delta=0.02)
        power, _ = derived["powers"]["qwen-desktop-b2"]
        self.assertAlmostEqual(power, COST.model("qwen").power_w(2, False), delta=1.0)

    def test_replay_and_policy_errors_are_bounded(self) -> None:
        replay = {row[0]: float(row[3].split()[0]) for row in evaluation.replay_error_rows(self.runs, self.totals, COST)}
        self.assertLess(abs(replay["s2a"]), 10.0)
        self.assertLess(abs(replay["legacy"]), 5.0)
        rows = evaluation.policy_error_rows(self.runs, self.totals, COST)
        s2a = next(row for row in rows if row[0] == "s2a" and row[1] == "mean loads")
        self.assertLess(abs(float(s2a[4].split()[0])), 6.0)
        legacy = next(row for row in rows if row[0] == "legacy" and row[1] == "mean loads")
        self.assertLess(abs(float(legacy[4].split()[0])), 3.0)

    def test_perturbed_instances_are_reproducible(self) -> None:
        requests = evaluation.trace_requests(self.runs["s2a"])
        first = list(evaluation.perturbed_instances(requests, self.runs, self.totals, 2, 7, 5.0))
        second = list(evaluation.perturbed_instances(requests, self.runs, self.totals, 2, 7, 5.0))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
