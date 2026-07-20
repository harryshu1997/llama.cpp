#!/usr/bin/env python3
"""Regressions for the S10-E2A v4 route-DAG repair."""

from __future__ import annotations

import copy
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import aggregate  # noqa: E402
import e2a_canon as canon  # noqa: E402
import resolver  # noqa: E402
from test_e2a import load_route  # noqa: E402


def context(slot=1):
    plan = canon.load_strict(FIXTURES / "plan.json")
    manifest = canon.load_strict(FIXTURES / "manifest.json")
    timeline = canon.load_strict(
        FIXTURES / f"timelines/tl.slot{slot:02d}.json")
    lifecycle = canon.load_strict(
        FIXTURES / f"lifecycle/lc.slot{slot:02d}.json")
    outcomes = canon.load_strict(
        FIXTURES / f"outcomes/outcomes.slot{slot:02d}.json")
    resolved = resolver.resolve_requests(
        manifest, outcomes, timeline, FIXTURES, f"slot{slot}", slot // 2)
    route = load_route(timeline["role"], resolved=False)
    return plan, manifest, timeline, lifecycle, resolved, route


def bind_route(plan, timeline, lifecycle, route):
    canon.seal(route)
    field = ("route_schedule_digest_control"
             if route["role"] == resolver.CONTROL_ROLE
             else "route_schedule_digest_treatment")
    plan[field] = route["record_sha256"]
    canon.seal(plan)
    timeline["route_schedule_digest"] = route["record_sha256"]
    for action in lifecycle["actions"]:
        action["route_schedule_digest"] = route["record_sha256"]
    canon.seal(lifecycle)


def resolve_route(plan, manifest, route):
    return resolver.resolve_route_schedule(
        route, plan, manifest, route["role"])


class RouteArtifact(unittest.TestCase):

    def test_route_digest_requires_the_resolved_record(self):
        plan, manifest, _timeline, _lifecycle, _resolved, route = context()
        route["route_id"] = "route.changed"
        canon.seal(route)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_route_schedule(
                route, plan, manifest, resolver.TREATMENT_ROLE)
        self.assertEqual(ctx.exception.code, "E_ROUTE_ARTIFACT")

    def test_route_role_and_manifest_are_plan_bound(self):
        plan, manifest, _timeline, _lifecycle, _resolved, route = context()
        route["role"] = resolver.CONTROL_ROLE
        canon.seal(route)
        plan["route_schedule_digest_treatment"] = route["record_sha256"]
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_route_schedule(
                route, plan, manifest, resolver.TREATMENT_ROLE)
        self.assertEqual(ctx.exception.code, "E_ROUTE_BINDING")

    def test_route_cycle_is_rejected(self):
        plan, manifest, timeline, lifecycle, _resolved, route = context()
        route["edges"].append({
            "edge_id": "e.result.h2d.cycle",
            "from_action_id": "a.result",
            "to_action_id": "a.h2d",
            "edge_kind": "DATA",
            "payload_id": "cycle.payload",
            "request_ids": ["req.0", "req.1", "req.2"],
        })
        bind_route(plan, timeline, lifecycle, route)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_route_schedule(
                route, plan, manifest, resolver.TREATMENT_ROLE)
        self.assertEqual(ctx.exception.code, "E_ROUTE_CYCLE")

    def test_result_requires_a_phone_d2h_ancestor(self):
        plan, manifest, timeline, lifecycle, _resolved, route = context()
        route["edges"] = [
            edge for edge in route["edges"]
            if edge["edge_id"] != "e.d2h.result"
        ]
        bind_route(plan, timeline, lifecycle, route)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_route_schedule(
                route, plan, manifest, resolver.TREATMENT_ROLE)
        self.assertEqual(ctx.exception.code, "E_PHONE_RESULT_PATH")

    def test_phone_assisted_requests_are_manifest_bound(self):
        plan, manifest, timeline, lifecycle, _resolved, route = context()
        route["phone_assisted_request_ids"].append("req.unknown")
        bind_route(plan, timeline, lifecycle, route)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_route_schedule(
                route, plan, manifest, resolver.TREATMENT_ROLE)
        self.assertEqual(ctx.exception.code, "E_ROUTE_REQUEST")


class LifecycleRouteMatch(unittest.TestCase):

    def test_lifecycle_rejects_an_unresolved_route_record(self):
        plan, _manifest, timeline, lifecycle, resolved, route = context()
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(
                lifecycle, timeline, plan["slots"][1], resolved, plan, route)
        self.assertEqual(ctx.exception.code, "E_UNRESOLVED_ROUTE")

    def test_unplanned_server_exec_is_rejected(self):
        plan, manifest, timeline, lifecycle, resolved, route = context()
        extra = copy.deepcopy(
            next(action for action in lifecycle["actions"]
                 if action["action_kind"] == "EXEC"))
        extra.update({
            "action_id": "a.server.real",
            "execution_domain": "SERVER",
            "device_identity": plan["server_device_ids"][0],
            "backend_kind": "CUDA",
        })
        lifecycle["actions"].append(extra)
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(
                lifecycle, timeline, plan["slots"][1], resolved, plan,
                resolve_route(plan, manifest, route))
        self.assertEqual(ctx.exception.code, "E_ROUTE_NODE_EXTRA")

    def test_resolved_route_is_immutable_after_validation(self):
        plan, manifest, timeline, lifecycle, resolved, route = context()
        route_evidence = resolve_route(plan, manifest, route)
        extra_action = copy.deepcopy(
            next(action for action in lifecycle["actions"]
                 if action["action_kind"] == "EXEC"))
        extra_action.update({
            "action_id": "a.server.after.validation",
            "execution_domain": "SERVER",
            "device_identity": plan["server_device_ids"][0],
            "backend_kind": "CUDA",
        })
        extra_node = copy.deepcopy(
            next(node for node in route["nodes"]
                 if node["action_kind"] == "EXEC"))
        extra_node.update({
            "action_id": extra_action["action_id"],
            "execution_domain": "SERVER",
            "device_identity": plan["server_device_ids"][0],
            "backend_kind": "CUDA",
        })
        route["nodes"].append(extra_node)
        lifecycle["actions"].append(extra_action)
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(
                lifecycle, timeline, plan["slots"][1], resolved, plan,
                route_evidence)
        self.assertEqual(ctx.exception.code, "E_ROUTE_NODE_EXTRA")

    def test_missing_phone_exec_is_rejected(self):
        plan, manifest, timeline, lifecycle, resolved, route = context()
        lifecycle["actions"] = [
            action for action in lifecycle["actions"]
            if action["action_id"] != "a.exec"
        ]
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(
                lifecycle, timeline, plan["slots"][1], resolved, plan,
                resolve_route(plan, manifest, route))
        self.assertEqual(ctx.exception.code, "E_ROUTE_NODE_MISSING")

    def test_phone_exec_requires_positive_duration(self):
        plan, manifest, timeline, lifecycle, resolved, route = context()
        execution = next(
            action for action in lifecycle["actions"]
            if action["action_id"] == "a.exec")
        execution["end_us"] = execution["start_us"]
        execution["ack_us"] = execution["start_us"]
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(
                lifecycle, timeline, plan["slots"][1], resolved, plan,
                resolve_route(plan, manifest, route))
        self.assertEqual(ctx.exception.code, "E_ACTION_DURATION")

    def test_route_edge_runtime_order_is_enforced(self):
        plan, manifest, timeline, lifecycle, resolved, route = context()
        d2h = next(
            action for action in lifecycle["actions"]
            if action["action_id"] == "a.d2h")
        result = next(
            action for action in lifecycle["actions"]
            if action["action_id"] == "a.result")
        result["enqueue_us"] = d2h["start_us"]
        result["start_us"] = d2h["start_us"]
        result["end_us"] = d2h["start_us"] + 1
        result["ack_us"] = d2h["start_us"] + 1
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(
                lifecycle, timeline, plan["slots"][1], resolved, plan,
                resolve_route(plan, manifest, route))
        self.assertEqual(ctx.exception.code, "E_ROUTE_EDGE_ORDER")

    def test_operator_island_is_route_bound(self):
        plan, manifest, timeline, lifecycle, resolved, route = context()
        execution = next(
            action for action in lifecycle["actions"]
            if action["action_id"] == "a.exec")
        execution["operator_island_digest"] = "f" * 64
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(
                lifecycle, timeline, plan["slots"][1], resolved, plan,
                resolve_route(plan, manifest, route))
        self.assertEqual(ctx.exception.code, "E_ROUTE_NODE_MISMATCH")

    def test_valid_server_continuation_path_passes(self):
        plan, manifest, timeline, lifecycle, resolved, route = context()
        requests = ["req.0", "req.1", "req.2"]
        route["edges"] = [
            edge for edge in route["edges"]
            if edge["edge_id"] != "e.d2h.result"
        ]
        route["nodes"].append({
            "action_id": "a.server.continue",
            "action_kind": "EXEC",
            "execution_domain": "SERVER",
            "device_identity": plan["server_device_ids"][0],
            "backend_kind": "CUDA",
            "operator_island_digest": canon.digest({
                "island": "server.synthetic.continuation",
            }),
            "model_digest": plan["model_digest"],
            "request_ids": requests,
            "input_requirement": "POSITIVE",
            "output_requirement": "POSITIVE",
            "duration_requirement": "POSITIVE",
            "lease_requirement": "REQUIRED",
        })
        route["edges"].extend([
            {
                "edge_id": "e.d2h.server",
                "from_action_id": "a.d2h",
                "to_action_id": "a.server.continue",
                "edge_kind": "DATA",
                "payload_id": "phone.result.activation",
                "request_ids": requests,
            },
            {
                "edge_id": "e.server.result",
                "from_action_id": "a.server.continue",
                "to_action_id": "a.result",
                "edge_kind": "DATA",
                "payload_id": "server.final.output",
                "request_ids": requests,
            },
        ])
        end = timeline["window_end_us"]
        continuation = {
            "action_id": "a.server.continue",
            "action_kind": "EXEC",
            "state": "COMPLETE",
            "enqueue_us": end - 1_500_000,
            "start_us": end - 1_490_000,
            "end_us": end - 1_460_000,
            "ack_us": end - 1_450_000,
            "lease_id": "lease.0",
            "parent_action_id": "a.d2h",
            "request_ids": requests,
            "route_schedule_digest": "",
            "execution_domain": "SERVER",
            "device_identity": plan["server_device_ids"][0],
            "backend_kind": "CUDA",
            "operator_island_digest":
                route["nodes"][-1]["operator_island_digest"],
            "model_digest": plan["model_digest"],
            "input_bytes": 4096,
            "output_bytes": 4096,
        }
        result = next(
            action for action in lifecycle["actions"]
            if action["action_id"] == "a.result")
        result["parent_action_id"] = "a.server.continue"
        lifecycle["actions"].append(continuation)
        bind_route(plan, timeline, lifecycle, route)
        route_evidence = resolver.resolve_route_schedule(
            route, plan, manifest, resolver.TREATMENT_ROLE)
        resolver.resolve_lifecycle(
            lifecycle, timeline, plan["slots"][1], resolved, plan,
            route_evidence)


class ProductionBoundary(unittest.TestCase):

    def test_production_still_refuses_external_anchor(self):
        with self.assertRaises((aggregate.AggregateError,
                                resolver.ResolveError)) as ctx:
            aggregate.evaluate(str(FIXTURES / "bundle.json"))
        self.assertEqual(ctx.exception.code, "E_ANCHOR_TRUST_ROOT")


if __name__ == "__main__":
    unittest.main()
