"""WS11 measured layer placement: cost model, profile store, exact / greedy solvers (brute-force checked)."""
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

from research_dev.scheduler._internal.layer_placement import (  # noqa: E402
    DESKTOP_CPU,
    DesktopPower,
    DeviceCostPriors,
    HelperDevice,
    LayerPlacementError,
    LayerPlacementProfile,
    ModelLayers,
    PlacementProblem,
    PowerOperatingPoint,
    evaluate_owners,
    ffn_layer_bytes,
    fit_desktop_power,
    layer_mask,
    layer_spec,
    mask_layers,
    placement_report,
    provisioning_needs,
    restrict_to_executable,
    solve_placement,
    tensor_bytes,
    with_envelope_caps,
)
from research_dev.scheduler._internal import layer_placement as core  # noqa: E402


def priors(rate=50e9, overhead=1.0, active_w=15.0, idle_w=0.5, transport="usb"):
    return DeviceCostPriors(
        bytes_per_s=MappingProxyType({"f16": rate, "q4": rate / 3}),
        overhead_ms=MappingProxyType({transport: MappingProxyType({1: overhead, 2: overhead * 1.2})}),
        rows_factor=MappingProxyType({1: 1.0, 2: 1.1}), active_marginal_w=active_w, idle_w=idle_w)


def profile(**devices) -> LayerPlacementProfile:
    return LayerPlacementProfile(desktop_power=DesktopPower(150.0, 50.0, 25.0, "measured"),
                                 desktop_bytes_per_s=25e9, device_priors=dict(devices))


def model(name="m", layers=range(8), size=500_000_000, **extra) -> ModelLayers:
    return ModelLayers(name, tuple(layers), MappingProxyType({"f16": size, "q4": size // 3}), **extra)


def device(name, *, formats=None, capacity=10**12, transport="usb", **extra) -> HelperDevice:
    return HelperDevice(name, MappingProxyType(formats or {"m": "f16"}), capacity, transport, **extra)


class FormatTests(unittest.TestCase):
    def test_layer_bytes_of_the_rig_models(self):
        self.assertEqual(ffn_layer_bytes(5120, 17408, "F16"), 534_773_760)
        self.assertEqual(ffn_layer_bytes(3840, 15360, "F16"), 353_894_400)
        self.assertEqual(ffn_layer_bytes(3840, 15360, "Q4_0"), 99_532_800)
        self.assertEqual(ffn_layer_bytes(5120, 17408, "Q4_K"), 150_405_120)
        self.assertEqual(ffn_layer_bytes(5120, 17408, "Q4_K", "Q4_K", "Q6_K"), 173_383_680)
        with self.assertRaises(LayerPlacementError):
            tensor_bytes("Q4_K", 10, 100)            # 100 columns do not fill a 256 block

    def test_masks_and_specs(self):
        self.assertEqual(layer_spec(mask_layers(layer_mask([0, 1, 2, 5, 7, 8]))), "0-2,5,7-8")
        self.assertEqual(layer_mask(range(18, 24)), 0xFC0000)
        with self.assertRaises(LayerPlacementError):
            layer_mask([64])


class ProfileTests(unittest.TestCase):
    def test_call_summaries_merge_call_weighted_and_round_trip(self):
        store = profile(phone=priors())
        store.observe_call_summary("phone", "m", "f16", "usb", 1, 100, 10.0, 9.0, "a")
        store.observe_call_summary("phone", "m", "f16", "usb", 1, 300, 14.0, 12.0, "b")
        stats = store.calls[core.CallKey("phone", "m", "f16", 1, "usb")]
        self.assertAlmostEqual(stats.rpc_ms, 13.0)
        self.assertAlmostEqual(stats.compute_ms, 11.25)
        self.assertEqual(stats.calls, 400)
        again = LayerPlacementProfile.from_json(json.loads(json.dumps(store.to_json())))
        self.assertEqual(again.to_json(), store.to_json())

    def test_forgetting_weights_recent_batches(self):
        store = profile(phone=priors())
        store.keep = 0.5
        store.observe_call_summary("phone", "m", "f16", "usb", 1, 100, 10.0, 9.0)
        store.observe_call_summary("phone", "m", "f16", "usb", 1, 100, 20.0, 19.0)
        stats = store.calls[core.CallKey("phone", "m", "f16", 1, "usb")]
        self.assertAlmostEqual(stats.rpc_ms, (10 * 50 + 20 * 100) / 150)
        self.assertEqual(stats.calls, 150)

    def test_provenance_ladder(self):
        store = profile(phone=priors(rate=50e9, overhead=2.0, transport="usb"))
        sizes = {"m": {"f16": 500_000_000}, "n": {"f16": 250_000_000}}
        # prior: bytes / rate + transport overhead
        cost = store.call_cost("phone", "m", "f16", "usb", 1, 500_000_000, sizes)
        self.assertEqual(cost.provenance, "prior")
        self.assertAlmostEqual(cost.compute_ms.value, 10.0)
        self.assertAlmostEqual(cost.rpc_ms.value, 12.0)
        # another model measured on the same device/format: scaled by bytes, overhead measured
        store.observe_call_summary("phone", "n", "f16", "usb", 1, 5000, 6.5, 6.0)
        cost = store.call_cost("phone", "m", "f16", "usb", 1, 500_000_000, sizes)
        self.assertEqual(cost.compute_ms.provenance, "scaled")
        self.assertAlmostEqual(cost.compute_ms.value, 12.0)
        self.assertAlmostEqual(cost.rpc_ms.value, 12.5)
        # exact measurement wins; another transport keeps the compute and swaps the overhead
        store.observe_call_summary("phone", "m", "f16", "usb", 1, 5000, 11.0, 10.5)
        self.assertEqual(store.call_cost("phone", "m", "f16", "usb", 1, 500_000_000, sizes).provenance, "prior")
        exact = store.call_cost("phone", "m", "f16", "usb", 1, 500_000_000, sizes)
        self.assertEqual((exact.rpc_ms.provenance, exact.rpc_ms.value), ("measured", 11.0))
        self.assertEqual(exact.rpc_ms.samples, 5000)
        store.device_priors["phone"] = replace(store.device_priors["phone"], overhead_ms=MappingProxyType({
            "usb": MappingProxyType({1: 2.0}), "wifi": MappingProxyType({1: 6.0})}))
        other = store.call_cost("phone", "m", "f16", "wifi", 1, 500_000_000, sizes)
        self.assertAlmostEqual(other.compute_ms.value, 10.5)
        self.assertAlmostEqual(other.rpc_ms.value, 16.5)
        # energy: measured per call, then per compute-ms of the device, then the active power prior
        store.observe_call_energy("phone", "n", 1, 60.0, 1000)
        energy = store.call_cost("phone", "m", "f16", "usb", 1, 500_000_000, sizes).energy_j
        self.assertEqual(energy.provenance, "scaled")
        self.assertAlmostEqual(energy.value, 0.06 / 6.0 * 10.5)

    def test_desktop_rows_extrapolate_and_prior_from_bytes(self):
        store = profile()
        store.desktop_rows_factor = MappingProxyType({1: 1.0, 2: 1.1})
        self.assertEqual(store.desktop_layer_ms("m", 1, 500_000_000).provenance, "prior")
        self.assertAlmostEqual(store.desktop_layer_ms("m", 1, 500_000_000).value, 20.0)
        store.observe_desktop_layer("m", 1, 18.0)
        self.assertAlmostEqual(store.desktop_layer_ms("m", 4, 500_000_000).value, 18.0 * 1.3)

    def test_fit_desktop_power_recovers_state_powers(self):
        truth = (150.0, 40.0, 30.0)
        points = []
        for ffn, wait, other in ((0.4, 0.0, 0.2), (0.02, 0.25, 0.2), (0.3, 0.0, 0.1), (0.04, 0.17, 0.18)):
            points.append(PowerOperatingPoint(ffn, wait, other, ffn * truth[0] + wait * truth[1] + other * truth[2]))
        power = fit_desktop_power(points)
        for value, expected in zip((power.ffn_w, power.wait_w, power.other_w), truth):
            self.assertAlmostEqual(value, expected, places=6)
        self.assertEqual(power.provenance, "derived")


def brute_force(problem: PlacementProblem):
    """Every assignment of every CPU layer to every allowed owner (tiny instances only)."""
    items = [item for row in problem.models for item in core._items_for(problem, row, 1.0)]
    limits = core._constraints(problem, items)
    best = None
    for choice in itertools.product(*(range(len(item.options)) for item in items)):
        if not core._feasible(problem, items, choice, limits):
            continue
        value = core._energy(items, choice)
        if best is None or value < best - 1e-12:
            best = value
    return best


def plan_energy(problem, plan):
    items, choice = [], []
    for row in problem.models:
        for item in core._items_for(problem, row, 1.0):
            items.append(item)
            owner = plan.owner(row.model_id, item.layer)
            choice.append(next(i for i, o in enumerate(item.options) if o.owner == owner))
    return core._energy(items, choice)


class SolverTests(unittest.TestCase):
    def random_problem(self, seed):
        rng = random.Random(seed)
        devices = {}
        store = profile()
        helpers = []
        for index in range(rng.choice((1, 2))):
            name = f"phone{index}"
            store.device_priors[name] = priors(rate=rng.uniform(15e9, 70e9), overhead=rng.uniform(0.3, 6.0),
                                               active_w=rng.uniform(3, 20))
            residency = rng.choice(("co-resident", "per-model"))
            sessions = rng.choice((1, 2))
            helpers.append(device(name, formats={"a": rng.choice(("f16", "q4")), "b": "f16"},
                                  capacity=rng.choice((600_000_000, 1_500_000_000, 10**12)), residency=residency,
                                  sessions=sessions, session_limit_bytes=rng.choice((0, 800_000_000)),
                                  max_busy_ms_per_token=rng.choice((None, 25.0)), transport="usb"))
            devices[name] = helpers[-1]
        models = (model("a", range(rng.randint(1, 3)), size=rng.choice((300_000_000, 500_000_000))),
                  model("b", range(rng.randint(1, 3)), size=400_000_000))
        return PlacementProblem(models, tuple(helpers), store,
                                rows_mix=MappingProxyType({"a": {1: 0.7, 2: 0.3}}),
                                latency_ppm=rng.choice((None, 1_000_000, 1_200_000)))

    def test_exact_matches_brute_force_and_greedy_respects_its_bound(self):
        offloaded = 0
        for seed in range(60):
            problem = self.random_problem(seed)
            best = brute_force(problem)
            cpu_only = core._energy([item for row in problem.models for item in core._items_for(problem, row, 1.0)],
                                    [0] * sum(len(row.cpu_layers) for row in problem.models))
            offloaded += best < cpu_only - 1e-9
            exact = solve_placement(problem, method="exact")
            self.assertAlmostEqual(plan_energy(problem, exact), best, places=9, msg=f"seed {seed}")
            greedy = solve_placement(problem, method="greedy")
            gap = plan_energy(problem, greedy) - best
            self.assertGreaterEqual(gap, -1e-9, msg=f"seed {seed}")
            self.assertLessEqual(gap, greedy.gap_bound_j_per_step + 1e-9, msg=f"seed {seed}")
        self.assertGreater(offloaded, 30)            # the instances exercise helpers, not just the CPU

    def test_capacity_session_packing_and_per_model_scopes(self):
        store = profile(phone=priors(rate=60e9, overhead=0.5))
        # 3 sessions of 1.1 GB hold 2 x 0.5 GB layers each -> 6 layers, not 6.6
        helper = device("phone", formats={"m": "f16", "n": "f16"}, capacity=3_300_000_000, sessions=3,
                        session_limit_bytes=1_100_000_000, residency="per-model")
        problem = PlacementProblem((model("m", range(10)), model("n", range(10))), (helper,), store)
        plan = solve_placement(problem)
        for name in ("m", "n"):          # per-model residency: each model gets the full capacity
            self.assertEqual(bin(plan.helper_masks(name).get("phone", 0)).count("1"), 6)
        shared = replace(problem, devices=(replace(helper, residency="co-resident"),))
        plan = solve_placement(shared)
        self.assertEqual(sum(bin(plan.helper_masks(name).get("phone", 0)).count("1") for name in ("m", "n")), 6)

    def test_slow_helper_gets_a_column_split_under_a_tight_latency_bound(self):
        # helper call 15 ms vs CPU 20 ms per layer: cheaper energy but a full offload is also fine; make the
        # helper SLOWER (30 ms) and cheap, with a 1.0x latency bound: full offload is infeasible, a 50 %
        # split runs in parallel with the host (max(10, 1 + 15) = 16 ms < 20 ms)
        store = profile(phone=priors(rate=500_000_000 / 0.029, overhead=1.0, active_w=1.0))
        problem = PlacementProblem((model("m", range(4)),), (device("phone"),), store, latency_ppm=1_000_000,
                                   column_fractions=MappingProxyType({"m": (1.0, 0.5)}))
        plan = solve_placement(problem)
        self.assertEqual(plan.models["m"].column_fraction, 0.5)
        self.assertEqual(plan.helper_masks("m"), {"phone": 0b1111})
        full = solve_placement(replace(problem, column_fractions=MappingProxyType({"m": (1.0,)})))
        self.assertEqual(full.helper_masks("m"), {})        # full offload breaks the 1.0x bound

    def test_busy_cap_and_envelope_growth_limit_a_device(self):
        store = profile(phone=priors(rate=60e9, overhead=0.5))
        helper = device("phone", busy_envelope_ms=20.0)
        problem = PlacementProblem((model("m", range(8)),), (helper,), store)
        self.assertEqual(bin(solve_placement(problem).helper_masks("m")["phone"]).count("1"), 8)
        capped = with_envelope_caps(problem, 250_000)      # <= 25 ms of ~8.8 ms calls -> 2 layers
        self.assertEqual(bin(solve_placement(capped).helper_masks("m")["phone"]).count("1"), 2)

    def test_unavailable_device_gets_nothing_and_current_is_priced_after_mask_out(self):
        store = profile(a=priors(rate=60e9, overhead=0.5), b=priors(rate=40e9, overhead=0.5))
        problem = PlacementProblem((model("m", range(4)),), (device("a", available=False), device("b")), store,
                                   current=MappingProxyType({"m": {0: "a", 1: "a", 2: "b", 3: DESKTOP_CPU}}))
        plan = solve_placement(problem)
        self.assertNotIn("a", plan.helper_masks("m"))
        effective = evaluate_owners(problem, {"m": {0: DESKTOP_CPU, 1: DESKTOP_CPU, 2: "b", 3: DESKTOP_CPU}})["m"]
        self.assertAlmostEqual(plan.models["m"].current_energy_j_per_token, effective)

    def test_equal_energy_keeps_current_owners_and_contiguous_ranges(self):
        store = profile(a=priors(rate=50e9, overhead=1.0), b=priors(rate=50e9, overhead=1.0))
        current = {layer: ("a" if layer < 3 else "b") for layer in range(6)}
        problem = PlacementProblem((model("m", range(6)),), (device("a", capacity=1_500_000_000), device("b")), store,
                                   current=MappingProxyType({"m": current}))
        plan = solve_placement(problem)
        self.assertEqual(plan.helper_masks("m"), {"a": 0b000111, "b": 0b111000})

    def test_risk_premium_prefers_the_well_measured_device(self):
        store = profile(a=priors(rate=50e9, overhead=1.0), b=priors(rate=50e9, overhead=1.0))
        store.observe_call_summary("a", "m", "f16", "usb", 1, 50_000, 11.0, 10.0)
        store.observe_call_summary("b", "m", "f16", "usb", 1, 300, 10.4, 9.6)     # cheaper, thinly measured
        base = PlacementProblem((model("m", range(2)),), (device("a", capacity=10**12), device("b")), store)
        self.assertEqual(set(solve_placement(base).helper_masks("m")), {"b"})
        risky = replace(base, uncertainty_ppm=MappingProxyType({"scaled": 100_000}), confident_calls=2000)
        self.assertEqual(set(solve_placement(risky).helper_masks("m")), {"a"})

    def test_executable_plan_and_provisioning_needs(self):
        store = profile(phone=priors(rate=60e9, overhead=0.5))
        helper = device("phone", stored_layers=MappingProxyType({"m": 0b0011}),
                        qualified_layers=MappingProxyType({"m": 0b0111}))
        problem = PlacementProblem((model("m", range(4)),), (helper,), store)
        report = placement_report(problem)
        self.assertEqual(report["ideal"]["models"]["m"]["owners"]["phone"]["layers"], "0-3")
        self.assertEqual(report["executable"]["models"]["m"]["owners"]["phone"]["layers"], "0-1")
        needs = provisioning_needs(problem, solve_placement(problem))
        self.assertEqual([(row.layers, row.needs_shard, row.needs_qualification) for row in needs], [((2, 3), True, True)])
        self.assertEqual(needs[0].bytes, 2 * 500_000_000)
        restricted = restrict_to_executable(problem)
        self.assertEqual(restricted.devices[0].allowed_layers["m"], 0b0011)

    def test_greedy_runs_large_instances_and_reports_a_bound(self):
        store = profile(**{f"p{i}": priors(rate=30e9 + i * 7e9, overhead=0.5 + i) for i in range(4)})
        helpers = tuple(device(f"p{i}", capacity=(i + 2) * 2_000_000_000) for i in range(4))
        problem = PlacementProblem((model("m", range(40)),), helpers, store)
        plan = solve_placement(problem, method="greedy")
        self.assertEqual(plan.solver, "greedy")
        problem = PlacementProblem((model("m", range(20)),), helpers, store)
        plan = solve_placement(problem, method="greedy")
        self.assertEqual(plan.solver, "greedy")
        self.assertGreaterEqual(plan.gap_bound_j_per_step, 0.0)
        exact = solve_placement(problem, method="auto")
        self.assertLessEqual(exact.models["m"].objective_j_per_token, plan.models["m"].objective_j_per_token + 1e-9)


if __name__ == "__main__":
    unittest.main()
