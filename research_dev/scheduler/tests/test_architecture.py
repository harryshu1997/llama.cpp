#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import unittest

import research_dev.scheduler as public

from test_runtime_controller import (
    model,
    profile,
    request,
    snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAY_RUNNER = REPO_ROOT / (
    "research_dev/spikes/s42_general_energy_scheduler_v1/"
    "full_fp16_burstgpt_v1/small_model_overlay_v1/"
    "run_fp16_small_overlay.py"
)
LARGE_RUNNER = REPO_ROOT / (
    "research_dev/spikes/s41_gemma_qwen_continuous_baseline/"
    "tp_operator_split_v1/mixed_scheduler_v1/run_hierarchical_trace.py"
)
UNIFIED_RUNNER = REPO_ROOT / (
    "research_dev/scheduler/campaigns/burstgpt/runner.py"
)
PHYSICAL_PREFLIGHT_RUNNER = REPO_ROOT / (
    "research_dev/scheduler/campaigns/burstgpt/preflight.py"
)
CANONICAL_PHYSICAL_RIG = REPO_ROOT / (
    "research_dev/scheduler/adapters/heterogeneous_rig.py"
)


class FakePhysicalAdapter:
    def __init__(self) -> None:
        observation = public.RuntimeExecutorObservation
        self.observations = (
            observation(
                executor_id="memory://host-executor",
                route_id="desktop-cpu",
                backend="cpu",
                resource_ids=("desktop-cpu",),
                memory_resource_id="host-ram",
                resident=True,
                health="healthy",
                qualification_facts={"shape_qualified": True},
                route_family="cpu",
            ),
            observation(
                executor_id="memory://helper-executor",
                route_id="phone-full",
                backend="accelerator",
                resource_ids=("phone-compute", "phone-link"),
                memory_resource_id="phone-ram",
                resident=True,
                health="healthy",
                qualification_facts={"thermal_qualified": False},
                route_family="phone",
            ),
        )
        self.executed = []

    def execute(self, binding: public.RuntimeExecutorBinding) -> None:
        self.executed.append(binding)


class RuntimeOwnershipTests(unittest.TestCase):
    def test_physical_rig_policy_is_canonical(self) -> None:
        campaign_tree = ast.parse(
            UNIFIED_RUNNER.read_text(encoding="ascii")
        )
        canonical = CANONICAL_PHYSICAL_RIG.read_text(encoding="ascii")
        campaign_definitions = {
            node.name for node in ast.walk(campaign_tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        self.assertNotIn("HeterogeneousPhysicalRig", campaign_definitions)
        self.assertIn("class HeterogeneousPhysicalRig", canonical)
        self.assertIn("CanonicalTransitionRegistry", canonical)
        self.assertNotIn("research_dev.spikes", canonical)

    def test_scheduler_adapters_import_without_spikes(self) -> None:
        script = """
import importlib.abc
import sys

class RejectSpikes(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('research_dev.spikes'):
            raise AssertionError(fullname)
        return None

sys.meta_path.insert(0, RejectSpikes())
from research_dev.scheduler.adapters import HeterogeneousPhysicalRig
assert HeterogeneousPhysicalRig.__module__.startswith(
    'research_dev.scheduler.adapters.'
)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stdout + completed.stderr,
        )

    def test_large_trace_runner_has_no_route_or_hold_policy(self) -> None:
        source = LARGE_RUNNER.read_text(encoding="ascii")
        tree = ast.parse(source)
        function_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue({
            "fp16_runtime_binding",
            "fp16_family_routes",
            "schedule_cpu_recovery",
            "schedule_fp16_online_request",
        }.isdisjoint(function_names))
        self.assertNotIn("held_rows", source)
        self.assertNotIn("hot_route", source)
        self.assertNotIn("cold_route", source)
        self.assertNotIn("RuntimeExecutorBinding", source)

    def test_unified_trace_runner_delegates_runtime_control(self) -> None:
        tree = ast.parse(UNIFIED_RUNNER.read_text(encoding="ascii"))
        constructed = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }
        self.assertIn("CanonicalArrivalCoordinator", constructed)
        self.assertIn("CanonicalRuntimeSubmission", constructed)
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue({
            "wait_runtime_request",
            "replan_automated_request",
            "fail_automated_request",
            "start_runtime_lease_renewal",
            "check_runtime_lease_renewal",
            "stop_runtime_lease_renewal",
            "complete_automated_request",
            "cancel_runtime_request",
        }.isdisjoint(called_attributes))

    def test_physical_preflight_delegates_readiness_policy(self) -> None:
        tree = ast.parse(
            PHYSICAL_PREFLIGHT_RUNNER.read_text(encoding="ascii")
        )
        constructed = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }
        self.assertIn(
            "materialize_preflight_executor_samples", constructed
        )
        self.assertNotIn("EndpointRuntimeSample", constructed)

    def test_scheduler_derives_eligibility_from_raw_observations(self) -> None:
        adapter = FakePhysicalAdapter()
        scheduler = public.UnifiedScheduler((profile(),), "enforce")
        ticket = scheduler.submit_runtime_request(
            request("raw-observation"),
            model(),
            adapter.observations,
            snapshot=snapshot(1_000),
            observed_at_us=1_000,
        )
        adapter.execute(ticket.binding)
        estimates = {
            row.route_id: row for row in ticket.cost_estimates.estimates
        }
        self.assertEqual(set(estimates), {"desktop-cpu", "phone-full"})
        self.assertFalse(estimates["phone-full"].admitted)
        self.assertEqual(
            ticket.binding.executor_id, "memory://host-executor"
        )
        self.assertEqual(adapter.executed, [ticket.binding])

    def test_scheduler_queues_qualified_route_behind_observed_phase(self) -> None:
        adapter = FakePhysicalAdapter()
        qualified = public.RuntimeExecutorObservation(
            executor_id="memory://qualified-helper",
            route_id="phone-full",
            backend="accelerator",
            resource_ids=("phone-compute", "phone-link"),
            memory_resource_id="phone-ram",
            resident=True,
            health="healthy",
            qualification_facts={"thermal_qualified": True},
            route_family="phone",
        )
        scheduler = public.UnifiedScheduler((profile(),), "enforce")
        scheduler.observe_runtime_phase(
            public.RuntimePhaseObservation(
                source_id="synthetic-protected-work",
                phase_id="protected",
                owner_id="protected-owner",
                phase_start_us=0,
                occupied_resource_ids=("phone-compute", "phone-link"),
            ),
            observed_at_us=1_000,
            renewal_us=5_000,
        )
        ticket = scheduler.submit_runtime_request(
            request("phase-queue"),
            model(),
            (adapter.observations[0], qualified),
            snapshot=snapshot(1_000),
            observed_at_us=1_000,
        )
        helper = next(
            row for row in ticket.cost_estimates.estimates
            if row.route_id == "phone-full"
        )
        self.assertTrue(helper.admitted)
        self.assertEqual(ticket.decision.route_id, "phone-full")
        self.assertGreaterEqual(ticket.decision.start_us, 6_000)

    def test_overlay_runner_contains_only_raw_runtime_observations(self) -> None:
        tree = ast.parse(OVERLAY_RUNNER.read_text(encoding="ascii"))
        function_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue({
            "phone_runtime_binding_ready",
            "split_runtime_binding_ready",
            "fallback_replan_allowed",
        }.isdisjoint(function_names))
        constructed = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("RuntimeExecutorBinding", constructed)
        self.assertNotIn("PhaseResourceManager", {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        })
        self.assertNotIn("S42EndpointBackend", {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        })

    def test_overlay_runner_delegates_the_complete_lifecycle(self) -> None:
        tree = ast.parse(OVERLAY_RUNNER.read_text(encoding="ascii"))
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue({
            "wait_runtime_request",
            "replan_automated_request",
            "replan_runtime_request",
            "fail_automated_request",
            "fail_runtime_request",
            "start_runtime_lease_renewal",
            "check_runtime_lease_renewal",
            "stop_runtime_lease_renewal",
            "complete_automated_request",
            "complete_runtime_request",
            "cancel_runtime_request",
        }.isdisjoint(called_attributes))
        constructed = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }
        self.assertIn("CanonicalPhysicalAdapter", constructed)
        self.assertIn("CanonicalHttpExecutionBackend", constructed)

    def test_overlay_runner_does_not_interpret_scheduler_execution_plans(
        self,
    ) -> None:
        tree = ast.parse(OVERLAY_RUNNER.read_text(encoding="ascii"))
        constructed = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertTrue({
            "RuntimeExecutionReceipt",
            "RuntimeTransitionReceipt",
            "RuntimeOperatorAssignment",
            "RuntimeTransitionPlan",
        }.isdisjoint(constructed))
        inspected = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        self.assertTrue({
            "operators",
            "split_fraction_ppm",
            "transitions",
            "transition_status",
            "execution_plan",
        }.isdisjoint(inspected))
        function_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue({
            "physical_split_contract",
            "validate_physical_split_ticket",
            "validate_physical_split_capability",
            "split_accelerator_prewarm",
        }.isdisjoint(function_names))
        self.assertTrue({
            "build_static_split_prewarm",
            "static_split_contract_values",
            "validate_static_split_capability",
        }.isdisjoint(imported))
        self.assertNotIn("partial", imported)
        self.assertIn("CanonicalStaticSplitPrewarmer", constructed)


if __name__ == "__main__":
    unittest.main()
