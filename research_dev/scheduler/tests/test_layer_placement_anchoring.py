"""WS12 runtime rules in measured placement: route anchors (co-helpers need the primary) and max models.

The runtime's assisted route belongs to the primary phone; a static co-helper only adds layers to it and
serves one model artifact per campaign. The planner must never emit a plan the runtime cannot run, and the
arm materializer must refuse one. Brute force here checks the rules independently of the solver's code.
"""
from __future__ import annotations

import itertools
import json
import random
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal import layer_placement as core  # noqa: E402
from research_dev.scheduler._internal.layer_placement import (  # noqa: E402
    DESKTOP_CPU,
    DesktopPower,
    DeviceCostPriors,
    HelperDevice,
    LayerPlacementError,
    LayerPlacementProfile,
    ModelLayers,
    PlacementProblem,
    restrict_to_executable,
    solve_placement,
)
from research_dev.scheduler._internal.layer_placement_io import (  # noqa: E402
    device_from_json,
    device_to_json,
)


def priors(rate=50e9, overhead=1.0, active_w=15.0, idle_w=0.5):
    return DeviceCostPriors(
        bytes_per_s=MappingProxyType({"f16": rate, "q4": rate / 3}),
        overhead_ms=MappingProxyType({"usb": MappingProxyType({1: overhead, 2: overhead * 1.2})}),
        rows_factor=MappingProxyType({1: 1.0, 2: 1.1}), active_marginal_w=active_w, idle_w=idle_w)


def profile(**devices) -> LayerPlacementProfile:
    return LayerPlacementProfile(desktop_power=DesktopPower(150.0, 50.0, 25.0, "measured"),
                                 desktop_bytes_per_s=25e9, device_priors=dict(devices))


def model(name, layers, size=500_000_000) -> ModelLayers:
    return ModelLayers(name, tuple(layers), MappingProxyType({"f16": size, "q4": size // 3}))


def device(name, formats, **extra) -> HelperDevice:
    return HelperDevice(name, MappingProxyType(formats), extra.pop("capacity", 10**12), "usb", **extra)


def owners(plan, model_id):
    return {owner: value["layers"] for owner, value in plan.models[model_id].to_json()["owners"].items()
            if value["count"]}


def runnable(problem: PlacementProblem, items, choice) -> bool:
    """The runtime rules, written independently of the solver."""
    by_model: dict[str, set[str]] = {}
    for item, index in zip(items, choice):
        by_model.setdefault(item.model_id, set()).add(item.options[index].owner)
    flags = {row.device_id: row for row in problem.devices}
    for helpers in by_model.values():
        helpers = helpers - {DESKTOP_CPU}
        if helpers and all(not flags[name].route_anchor for name in helpers):
            return False
    for name, row in flags.items():
        if row.max_models is not None and sum(name in value for value in by_model.values()) > row.max_models:
            return False
    return True


def brute_force(problem: PlacementProblem):
    """Every assignment; capacity/busy/latency from the solver's checks with the rules switched off there."""
    relaxed = replace(problem, devices=tuple(replace(row, route_anchor=True, max_models=None)
                                             for row in problem.devices))
    items = [item for row in relaxed.models for item in core._items_for(relaxed, row, 1.0)]
    limits = core._constraints(relaxed, items)
    best = None
    for choice in itertools.product(*(range(len(item.options)) for item in items)):
        if not core._feasible(relaxed, items, choice, limits) or not runnable(problem, items, choice):
            continue
        value = core._energy(items, choice)
        if best is None or value < best - 1e-12:
            best = value
    return best


def plan_choice(problem, plan):
    items, choice = [], []
    for row in problem.models:
        for item in core._items_for(problem, row, 1.0):
            items.append(item)
            owner = plan.owner(row.model_id, item.layer)
            choice.append(next(i for i, o in enumerate(item.options) if o.owner == owner))
    return items, choice


class AnchorTests(unittest.TestCase):
    def two_phone(self, **pixel):
        store = profile(primary=priors(rate=30e9, overhead=2.0), cohelper=priors(rate=70e9, overhead=0.3))
        devices = (device("primary", {"q": "f16", "g": "f16"}),
                   device("cohelper", {"q": "f16", "g": "f16"}, **pixel))
        return PlacementProblem((model("q", range(6)), model("g", range(4), size=300_000_000)), devices, store)

    def test_all_anchors_is_the_old_problem(self):
        problem = self.two_phone()
        self.assertFalse(problem.anchoring)
        plan = solve_placement(problem)
        # the co-helper is cheaper everywhere: without the rule it takes every layer of both models
        self.assertEqual(owners(plan, "q"), {"cohelper": "0-5"})
        self.assertEqual(owners(plan, "g"), {"cohelper": "0-3"})

    def test_a_co_helper_never_serves_a_model_alone(self):
        problem = self.two_phone(route_anchor=False)
        for method in ("exact", "greedy"):
            plan = solve_placement(problem, method=method)
            for name in ("q", "g"):
                held = owners(plan, name)
                self.assertIn("primary", held, (method, name, held))
                self.assertEqual(sum(1 for layer, owner in plan.models[name].owners.items() if owner == "primary"),
                                 1, (method, name, held))   # one anchor layer is enough, the rest stays cheap
            self.assertTrue(runnable(problem, *plan_choice(problem, plan)))
        self.assertAlmostEqual(core._energy(*plan_choice(problem, solve_placement(problem, method="exact"))),
                               brute_force(problem), places=9)

    def test_without_an_anchor_the_co_helper_gets_nothing(self):
        problem = self.two_phone(route_anchor=False)
        gone = replace(problem, devices=tuple(replace(row, available=row.device_id != "primary")
                                              for row in problem.devices))
        for method in ("exact", "greedy"):
            plan = solve_placement(gone, method=method)
            self.assertEqual(owners(plan, "q"), {DESKTOP_CPU: "0-5"})
            self.assertEqual(owners(plan, "g"), {DESKTOP_CPU: "0-3"})
        # executable plans inherit the rule: the anchor stores nothing for "g" -> the co-helper may not serve it
        stored = replace(problem, devices=(
            replace(problem.devices[0], stored_layers=MappingProxyType({"q": 0b1}),
                    qualified_layers=MappingProxyType({"q": 0b1})),
            replace(problem.devices[1], stored_layers=MappingProxyType({"q": 0b111110, "g": 0b1111}),
                    qualified_layers=MappingProxyType({"q": 0b111110, "g": 0b1111}))))
        plan = solve_placement(restrict_to_executable(stored))
        self.assertEqual(owners(plan, "q"), {"primary": "0", "cohelper": "1-5"})
        self.assertEqual(owners(plan, "g"), {DESKTOP_CPU: "0-3"})

    def test_max_models_keeps_a_co_helper_on_one_model(self):
        problem = self.two_phone(route_anchor=False, max_models=1)
        for method in ("exact", "greedy"):
            plan = solve_placement(problem, method=method)
            served = [name for name in ("q", "g") if "cohelper" in owners(plan, name)]
            self.assertEqual(len(served), 1, (method, served))
            self.assertTrue(runnable(problem, *plan_choice(problem, plan)))
        exact = solve_placement(problem, method="exact")
        self.assertAlmostEqual(core._energy(*plan_choice(problem, exact)), brute_force(problem), places=9)

    def test_random_instances_match_brute_force(self):
        constrained = 0
        for seed in range(80):
            rng = random.Random(1000 + seed)
            store = profile()
            devices = []
            for index in range(rng.choice((2, 3))):
                name = f"phone{index}"
                store.device_priors[name] = priors(rate=rng.uniform(15e9, 80e9), overhead=rng.uniform(0.2, 5.0),
                                                   active_w=rng.uniform(3, 20))
                devices.append(device(name, {"a": rng.choice(("f16", "q4")), "b": "f16"},
                                      capacity=rng.choice((900_000_000, 1_600_000_000, 10**12)),
                                      residency=rng.choice(("co-resident", "per-model")),
                                      route_anchor=index == 0 or rng.random() < 0.3,
                                      max_models=rng.choice((None, None, 1)),
                                      max_busy_ms_per_token=rng.choice((None, 30.0))))
            problem = PlacementProblem(
                (model("a", range(rng.randint(1, 3)), size=rng.choice((300_000_000, 500_000_000))),
                 model("b", range(rng.randint(1, 3)), size=400_000_000)),
                tuple(devices), store, rows_mix=MappingProxyType({"a": {1: 0.7, 2: 0.3}}),
                latency_ppm=rng.choice((None, 1_000_000, 1_200_000)))
            best = brute_force(problem)
            relaxed = replace(problem, devices=tuple(replace(row, route_anchor=True, max_models=None)
                                                     for row in problem.devices))
            constrained += best > brute_force(relaxed) + 1e-9
            exact = solve_placement(problem, method="exact")
            self.assertAlmostEqual(core._energy(*plan_choice(problem, exact)), best, places=9, msg=f"seed {seed}")
            greedy = solve_placement(problem, method="greedy")
            self.assertTrue(runnable(problem, *plan_choice(problem, greedy)), f"seed {seed}")
            gap = core._energy(*plan_choice(problem, greedy)) - best
            self.assertGreaterEqual(gap, -1e-9, msg=f"seed {seed}")
            self.assertLessEqual(gap, greedy.gap_bound_j_per_step + 1e-9, msg=f"seed {seed}")
        self.assertGreater(constrained, 10)    # the rules bind on a fair share of the instances

    def test_inventory_json_round_trip_and_validation(self):
        anchor = device("a", {"m": "f16"})
        self.assertNotIn("route_anchor", device_to_json(anchor))      # old inventories stay byte-identical
        self.assertNotIn("max_models", device_to_json(anchor))
        helper = device("h", {"m": "f16"}, route_anchor=False, max_models=1)
        row = device_to_json(helper)
        self.assertEqual((row["route_anchor"], row["max_models"]), (False, 1))
        self.assertEqual(device_from_json(json.loads(json.dumps(row))), helper)
        for bad in ({"route_anchor": "no"}, {"route_anchor": 0}, {"max_models": 0}, {"max_models": 1.0}):
            with self.assertRaises(LayerPlacementError):
                device_from_json({**row, **bad})
        with self.assertRaises(LayerPlacementError):
            device("x", {"m": "f16"}, max_models=0)


if __name__ == "__main__":
    unittest.main()
