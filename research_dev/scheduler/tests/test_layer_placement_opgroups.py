"""WS14 operator-group placement: GGUF geometry, option construction, brute-force exactness, byte identity."""
from __future__ import annotations

import itertools
import json
import random
import sys
import tempfile
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
    LayerPlacementProfile,
    ModelLayers,
    PlacementProblem,
    _constraints,
    _feasible,
    _usage_ok,
    solve_placement,
)
from research_dev.scheduler._internal import layer_placement_opgroups as og  # noqa: E402
from research_dev.scheduler._internal.layer_placement_io import device_from_json, device_to_json  # noqa: E402


def priors(rate=50e9, overhead=1.0, active_w=15.0, idle_w=0.5, transport="usb"):
    return DeviceCostPriors(
        bytes_per_s=MappingProxyType({"f16": rate, "pk": rate / 3}),
        overhead_ms=MappingProxyType({transport: MappingProxyType({1: overhead, 2: overhead * 1.2})}),
        rows_factor=MappingProxyType({1: 1.0, 2: 1.1}), active_marginal_w=active_w, idle_w=idle_w)


def profile(**devices) -> LayerPlacementProfile:
    return LayerPlacementProfile(desktop_power=DesktopPower(150.0, 50.0, 25.0, "measured"),
                                 desktop_bytes_per_s=25e9, device_priors=dict(devices))


def geometry(name="m", layers=6, *, proj=120_000_000, kv=4096, window_every=0, head=1_500_000_000):
    rows = []
    for index in range(layers):
        window = 1024 if window_every and index % window_every == 0 else None
        table = {"f16": MappingProxyType({"q": proj * 2 // 5, "k": proj // 10, "v": proj // 10, "o": proj * 2 // 5}),
                 "pk": MappingProxyType({"q": proj * 2 // 15, "k": proj // 30, "v": proj // 30, "o": proj * 2 // 15})}
        rows.append(og.LayerGeometry(index, 5120, 1024, 1024, 5120, MappingProxyType(table), kv, window))
    return og.ModelGeometry(name, "test", 5120, 151936, tuple(rows),
                            MappingProxyType({"f16": head, "pk": head // 3}))


def model(name="m", layers=range(4), size=500_000_000, other=100.0, **extra) -> ModelLayers:
    return ModelLayers(name, tuple(layers), MappingProxyType({"f16": size, "pk": size // 3}),
                       other_step_ms=MappingProxyType({1: other}), **extra)


def device(name, *, formats=None, capacity=10**12, transport="usb", **extra) -> HelperDevice:
    return HelperDevice(name, MappingProxyType(formats or {"m": "f16"}), capacity, transport, **extra)


def inputs(models=("m",), *, layers=6, context=500, slots=2, **extra) -> og.OpGroupInputs:
    return og.OpGroupInputs(
        geometry=MappingProxyType({name: geometry(name, layers) for name in models}),
        contexts=MappingProxyType({name: og.ContextStats.from_requests([(context, 200), (context // 2, 100)])
                                   for name in models}),
        slots=MappingProxyType({name: slots for name in models}),
        link_bytes_per_s=MappingProxyType({"usb": 200e6, "tcp": 100e6}), **extra)


def _brute_force(problem: PlacementProblem, opg: og.OpGroupInputs) -> float:
    """Minimum objective over every combination of UNPRUNED options (fraction 1), all limits checked."""
    adjusted, _ = og._adjusted(problem, opg)
    items = []
    for item_model in adjusted.models:
        for layer in sorted(item_model.cpu_layers):
            items.append(og._Item(item_model.model_id, layer,
                                  og.opgroup_options(adjusted, opg, item_model, layer, 1.0, prune=False)))
    limits = _constraints(adjusted, items)
    best = None
    for choice in itertools.product(*(range(len(item.options)) for item in items)):
        if not _feasible(adjusted, items, choice, limits):
            continue
        value = sum(item.options[index].energy_j for item, index in zip(items, choice))
        if best is None or value < best:
            best = value
    return best


def _plan_objective(plan: og.OpGroupPlan, problem: PlacementProblem) -> float:
    total = 0.0
    for model_id, row in plan.models.items():
        head = row["output_head"]["saving_vs_gpu_j_per_token"]
        total += (row["objective_j_per_token"] + head) * row["tokens_per_step"]
    return total


class GeometryTests(unittest.TestCase):
    def _write(self, path, *, arch, tied, v_missing_layer=None, window=None, pattern=None, dtype="f16"):
        sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
        import gguf
        import numpy as np

        writer = gguf.GGUFWriter(str(path), arch)
        writer.add_uint32(f"{arch}.block_count", 2)
        writer.add_uint32(f"{arch}.embedding_length", 64)
        if window:
            writer.add_uint32(f"{arch}.attention.sliding_window", window)
            writer.add_array(f"{arch}.attention.sliding_window_pattern", pattern)
        np_type = np.float16 if dtype == "f16" else np.float32
        for index in range(2):
            writer.add_tensor(f"blk.{index}.attn_q.weight", np.zeros((96, 64), dtype=np_type))
            writer.add_tensor(f"blk.{index}.attn_k.weight", np.zeros((32, 64), dtype=np_type))
            if index != v_missing_layer:
                writer.add_tensor(f"blk.{index}.attn_v.weight", np.zeros((32, 64), dtype=np_type))
            writer.add_tensor(f"blk.{index}.attn_output.weight", np.zeros((64, 96), dtype=np_type))
        writer.add_tensor("token_embd.weight", np.zeros((1000, 64), dtype=np_type))
        if not tied:
            writer.add_tensor("output.weight", np.zeros((1000, 64), dtype=np_type))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return gguf

    def test_header_reader_and_geometry_widths_bytes_kv_window_head(self):
        with tempfile.TemporaryDirectory() as directory:
            f16 = Path(directory) / "m-f16.gguf"
            f32 = Path(directory) / "m-f32.gguf"
            gguf = self._write(f16, arch="testarch", tied=False)
            self._write(f32, arch="testarch", tied=False, dtype="f32")
            header = og.read_gguf_header(f16, gguf)
            self.assertEqual(header["tensors"]["blk.0.attn_q.weight"], {"type": "F16", "shape": [64, 96]})
            geo = og.model_geometry("m", header, {"f16": header, "f32": og.read_gguf_header(f32, gguf)})
            layer = geo.layer(0)
            self.assertEqual((layer.q_out, layer.k_out, layer.v_out, layer.o_in), (96, 32, 32, 96))
            self.assertEqual(layer.proj_bytes["f16"]["q"], 96 * 64 * 2)
            self.assertEqual(layer.proj_bytes["f32"]["o"], 96 * 64 * 4)
            self.assertEqual(layer.kv_bytes_per_token, (32 + 32) * 2)
            self.assertIsNone(layer.window)
            self.assertEqual((geo.n_vocab, geo.head_tied, geo.head_bytes["f16"]), (1000, False, 1000 * 64 * 2))
            again = og.ModelGeometry.from_json(json.loads(json.dumps(geo.to_json())))
            self.assertEqual(again.to_json(), geo.to_json())

    def test_sliding_window_derived_v_and_tied_head(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g.gguf"
            gguf = self._write(path, arch="gemmatest", tied=True, v_missing_layer=1, window=16, pattern=[True, False])
            header = og.read_gguf_header(path, gguf)
            geo = og.model_geometry("g", header, {"f16": header})
            self.assertEqual(geo.layer(0).window, 16)
            self.assertIsNone(geo.layer(1).window)
            # no attn_v: V is derived from K but cached separately -> the cache still holds 2 x K width
            self.assertEqual((geo.layer(1).v_out, geo.layer(1).kv_bytes_per_token), (0, (32 + 32) * 2))
            self.assertNotIn("v", geo.layer(1).proj_bytes["f16"])
            self.assertTrue(geo.head_tied)
            self.assertEqual(geo.head_bytes["f16"], 1000 * 64 * 2)

    def test_recorded_rig_geometry_matches_the_models(self):
        from research_dev.scheduler.campaigns.burstgpt import layer_placement_rig as rig
        geo = rig.model_geometry_v1()
        qwen, gemma = geo[rig.QWEN], geo[rig.GEMMA]
        self.assertEqual(sum(qwen.layer(3).proj_bytes["f16"].values()), 125_829_120)     # 125.8 MB per layer
        self.assertEqual(qwen.layer(3).kv_bytes_per_token, 4096)
        self.assertEqual(qwen.head_bytes["f16"], 1_555_824_640)
        self.assertEqual((gemma.layer(0).window, gemma.layer(0).kv_bytes_per_token), (1024, 8192))
        self.assertEqual((gemma.layer(5).window, gemma.layer(5).v_out, gemma.layer(5).kv_bytes_per_token),
                         (None, 0, 2048))
        self.assertTrue(gemma.head_tied)
        stats, slots = rig.trace_contexts_v1()
        self.assertEqual((stats[rig.QWEN].decode_tokens, stats[rig.GEMMA].decode_tokens), (1524, 2034))
        self.assertEqual(slots, {rig.QWEN: 4, rig.GEMMA: 2})


class ContextTests(unittest.TestCase):
    def test_mean_decode_context_counts_generated_tokens(self):
        stats = og.ContextStats.from_requests([(100, 4), (0, 2)])
        # request 1: 100,101,102,103 ; request 2: 0,1 -> 407 / 6
        self.assertAlmostEqual(stats.mean_decode_context, 407 / 6)
        self.assertEqual(stats.max_context, 104)
        self.assertAlmostEqual(stats.prompt_tokens_per_decoded, 100 / 6)


class KvCapacityTests(unittest.TestCase):
    def test_kv_shares_capacity_but_not_sessions(self):
        from research_dev.scheduler._internal.layer_placement import _Constraints
        limits = _Constraints(MappingProxyType({("d", "m"): 1000}), MappingProxyType({("d", "m"): (2, 400)}),
                              MappingProxyType({}), MappingProxyType({}))
        self.assertTrue(_usage_ok({("d", "m"): [400, 400], ("d", "m+kv"): [200]}, limits))
        self.assertFalse(_usage_ok({("d", "m"): [400, 400], ("d", "m+kv"): [201]}, limits))     # capacity shared
        self.assertTrue(_usage_ok({("d", "m+kv"): [1000]}, limits))                                # KV only
        self.assertFalse(_usage_ok({("d", "m"): [401]}, limits))                                   # session limit


class OptionTests(unittest.TestCase):
    def setUp(self):
        self.profile = profile(phone=priors(), other=priors(rate=30e9, overhead=3.0))
        self.inputs = inputs()

    def problem(self, kernels, *, fractions=None, **device_extra):
        devices = (device("phone", kernels=frozenset(kernels), **device_extra),)
        return PlacementProblem((model(),), devices, self.profile,
                                column_fractions=MappingProxyType(fractions or {}))

    def test_ffn_only_options_are_ws11_plus_the_host_projection_and_kv_time(self):
        problem = self.problem({"ffn"})
        adjusted, _ = og._adjusted(problem, self.inputs)
        from research_dev.scheduler._internal.layer_placement import _option_costs
        base = _option_costs(adjusted, adjusted.models[0], 1, 1.0)
        wide = og.opgroup_options(adjusted, self.inputs, adjusted.models[0], 1, 1.0)
        host = og._host_costs(adjusted, self.inputs, adjusted.models[0], 1)
        self.assertEqual([o.owner for o in wide], [o.owner for o in base])
        for a, b in zip(base, wide):
            self.assertAlmostEqual(b.ms_by_rows[1] - a.ms_by_rows[1], host.proj_ms[1] + host.kv_ms[1])
            self.assertAlmostEqual(b.expected_j - a.expected_j,
                                   (150.0 + 0.5) * (host.proj_ms[1] + host.kv_ms[1]) / 1000.0)
            self.assertEqual((b.variant, b.parts), ("", ()))

    def test_fused_needs_the_kernel_full_width_and_the_ffn_owner(self):
        model_row = self.problem({"ffn"}).models[0]
        none = og.opgroup_options(*og._adjusted(self.problem({"ffn"}), self.inputs)[:1], self.inputs, model_row, 1, 1.0)
        self.assertFalse(any(o.variant.startswith("proj") for o in none))
        problem, _ = og._adjusted(self.problem({"ffn", "attn_proj:fused"}), self.inputs)
        fused = [o for o in og.opgroup_options(problem, self.inputs, problem.models[0], 1, 1.0)
                 if o.variant == "proj-fused"]
        self.assertTrue(fused and all(o.owner == "phone" and not o.parts for o in fused))
        half = og.opgroup_options(problem, self.inputs, problem.models[0], 1, 0.5)
        self.assertFalse(any(o.variant.startswith("proj-fused") for o in half))
        # the unit of layer 1 is O(1) + Q/K/V(2); the last CPU layer carries only its own O
        geo = self.inputs.geometry["m"]
        self.assertEqual(og._unit_bytes(geo, 1, "f16", ("o",), ("q", "k", "v")), 120_000_000)
        own, nxt = og._proj_unit(problem.models[0], geo, 3)
        self.assertEqual((own, nxt), (("o",), ()))

    def test_separate_projection_on_a_cpu_owned_layer_uses_the_helper_capacity(self):
        problem, _ = og._adjusted(self.problem({"ffn", "attn_proj:separate"}), self.inputs)
        options = og.opgroup_options(problem, self.inputs, problem.models[0], 1, 1.0, prune=False)
        cpu_sep = [o for o in options if o.owner == DESKTOP_CPU and o.variant == "proj-separate"]
        self.assertEqual(len(cpu_sep), 1)
        self.assertEqual(cpu_sep[0].parts[0][0], "phone")
        self.assertEqual(cpu_sep[0].parts[0][1], 120_000_000)
        # two calls (O after attention, Q/K/V before the next) -> two overheads in the separate time
        self.assertGreater(cpu_sep[0].parts[0][2], 2 * 1.0)

    def test_kv_variant_carries_kv_memory_and_is_never_on_a_co_helper_without_fused_projections(self):
        problem, _ = og._adjusted(self.problem({"ffn", "attn_kv:fused"}), self.inputs)
        self.assertFalse(any(o.variant == "proj-fused+kv"
                             for o in og.opgroup_options(problem, self.inputs, problem.models[0], 1, 1.0)))
        problem, _ = og._adjusted(self.problem({"ffn", "attn_proj:fused", "attn_kv:fused"}), self.inputs)
        kv = [o for o in og.opgroup_options(problem, self.inputs, problem.models[0], 1, 1.0, prune=False)
              if o.variant == "proj-fused+kv"]
        self.assertEqual(len(kv), 1)
        (device_id, size, busy, kind), = kv[0].parts
        stats = self.inputs.contexts["m"]
        self.assertEqual((device_id, kind, busy), ("phone", "kv", 0.0))
        self.assertEqual(size, int(2 * stats.max_context * 4096))

    def test_kv_bytes_grow_per_row_and_window_caps_them(self):
        problem, _ = og._adjusted(self.problem({"ffn"}), self.inputs)
        problem = replace(problem, rows_mix=MappingProxyType({"m": MappingProxyType({1: 0.5, 2: 0.5})}))
        host = og._host_costs(problem, self.inputs, problem.models[0], 1)
        self.assertAlmostEqual(host.kv_bytes_by_rows[2], 2 * host.kv_bytes_by_rows[1])
        capped = replace(self.inputs, geometry=MappingProxyType({"m": geometry("m", window_every=1)}),
                         contexts=MappingProxyType({"m": og.ContextStats.constant(100_000)}))
        self.assertEqual(og._host_costs(problem, capped, problem.models[0], 1).kv_bytes_by_rows[1], 1024 * 4096)


class ExactnessTests(unittest.TestCase):
    """Random small instances: the solver's objective equals brute force over every unpruned option."""

    def random_problem(self, rng: random.Random):
        n_layers = rng.randint(2, 4)
        anchor = rng.random() < 0.8
        kernels_a = frozenset({"ffn"} | set(rng.sample(["attn_proj:fused", "attn_proj:separate", "attn_kv:fused"],
                                                       rng.randint(0, 3))))
        kernels_b = frozenset({"ffn"} | set(rng.sample(["attn_proj:fused", "attn_proj:separate"], rng.randint(0, 2))))
        cap_a = rng.choice([10**12, 1_300_000_000, 900_000_000])
        devices = [device("a", formats={"m": "f16"}, capacity=cap_a, kernels=kernels_a,
                          **({"sessions": 2, "session_limit_bytes": rng.choice([700_000_000, 1_200_000_000]),
                              "residency": "per-model"} if rng.random() < 0.5 else {})),
                   device("b", formats={"m": "pk"}, capacity=rng.choice([10**12, 400_000_000]), transport="tcp",
                          kernels=kernels_b, route_anchor=anchor or rng.random() < 0.5,
                          max_busy_ms_per_token=rng.choice([None, 25.0]))]
        if not anchor:
            devices[0] = replace(devices[0], route_anchor=True)
        store = profile(a=priors(rate=rng.choice([40e9, 60e9]), overhead=rng.choice([0.5, 2.0])),
                        b=priors(rate=rng.choice([20e9, 40e9]), overhead=rng.choice([1.0, 5.0]), transport="tcp"))
        problem = PlacementProblem((model(layers=range(n_layers), other=rng.choice([20.0, 200.0])),), tuple(devices),
                                   store, latency_ppm=rng.choice([None, 1_000_000, 1_100_000]),
                                   rows_mix=MappingProxyType({"m": MappingProxyType({1: 0.7, 2: 0.3})}))
        opg = inputs(layers=n_layers + 1, context=rng.choice([200, 3000, 60000]), slots=rng.choice([1, 4]))
        return problem, opg

    def test_random_instances_match_brute_force(self):
        rng = random.Random(20261001)
        checked = 0
        for _ in range(60):
            problem, opg = self.random_problem(rng)
            plan = og.solve_op_groups(problem, opg, method="exact")
            expected = _brute_force(problem, opg)
            # plan JSON rounds J/token to 1e-6: allow that rounding times the tokens per step
            self.assertAlmostEqual(_plan_objective(plan, problem), expected, delta=5e-6)
            for layer in plan.models["m"]["layers"]:
                if layer["variant"].startswith("proj-fused"):
                    self.assertNotEqual(layer["ffn"], DESKTOP_CPU)      # fused only on the FFN owner
            checked += 1
        self.assertEqual(checked, 60)

    def test_relaxation_bound_never_exceeds_the_optimum(self):
        rng = random.Random(99)
        for _ in range(40):
            problem, opg = self.random_problem(rng)
            adjusted, _ = og._adjusted(problem, opg)
            items = [og._Item(m.model_id, layer, og.opgroup_options(adjusted, opg, m, layer, 1.0))
                     for m in adjusted.models for layer in sorted(m.cpu_layers)]
            limits = _constraints(adjusted, items)
            choice, bound = og._bound_dp(adjusted, items, limits)
            optimum = _brute_force(problem, opg)
            self.assertLessEqual(bound, optimum + 1e-9)
            if _feasible(adjusted, items, choice, limits):
                self.assertAlmostEqual(bound, optimum, places=9)

    def test_more_kernels_never_make_the_plan_worse(self):
        rng = random.Random(4242)
        for _ in range(25):
            problem, opg = self.random_problem(rng)
            base = replace(problem, devices=tuple(replace(d, kernels=frozenset({"ffn"})) for d in problem.devices))
            wide = replace(problem, devices=tuple(replace(d, kernels=og.ALL_KERNELS) for d in problem.devices))
            for method in ("auto", "greedy"):
                small = og.solve_op_groups(base, opg, method=method)
                large = og.solve_op_groups(wide, opg, method=method)
                self.assertLessEqual(large.models["m"]["objective_j_per_token"],
                                     small.models["m"]["objective_j_per_token"] + 1e-6)

    def test_greedy_is_feasible_and_within_its_gap_bound(self):
        rng = random.Random(7)
        for _ in range(20):
            problem, opg = self.random_problem(rng)
            greedy = og.solve_op_groups(problem, opg, method="greedy")
            exact = og.solve_op_groups(problem, opg, method="exact")
            tokens = greedy.models["m"]["tokens_per_step"]
            loss = (greedy.models["m"]["objective_j_per_token"] - exact.models["m"]["objective_j_per_token"]) * tokens
            self.assertGreaterEqual(loss, -1e-9)
            self.assertLessEqual(loss, greedy.gap_bound_j_per_step + 1e-6)


class IdentityTests(unittest.TestCase):
    def test_ffn_only_plan_is_the_ws11_plan(self):
        store = profile(a=priors(), b=priors(rate=20e9, overhead=4.0, transport="tcp"))
        problem = PlacementProblem((model(layers=range(6)),),
                                   (device("a", capacity=1_600_000_000), device("b", formats={"m": "pk"}, transport="tcp",
                                                                                 capacity=600_000_000)),
                                   store, latency_ppm=1_250_000,
                                   column_fractions=MappingProxyType({"m": (1.0, 0.5)}))
        ws11 = solve_placement(problem)
        plan = og.solve_op_groups(problem, inputs(layers=7))
        owners = {row["layer"]: row["ffn"] for row in plan.models["m"]["layers"]}
        self.assertEqual(owners, dict(ws11.models["m"].owners))
        self.assertEqual(plan.models["m"]["column_fraction"], ws11.models["m"].column_fraction)
        adjusted, _ = og._adjusted(problem, inputs(layers=7))
        host = sum(og._host_costs(adjusted, inputs(layers=7), adjusted.models[0], layer).proj_ms[1]
                   + og._host_costs(adjusted, inputs(layers=7), adjusted.models[0], layer).kv_ms[1] for layer in range(6))
        self.assertAlmostEqual(plan.models["m"]["energy_j_per_token"] - ws11.models["m"].energy_j_per_token,
                               (150.0 + 1.0) * host / 1000.0, places=5)

    def test_kernels_are_optional_in_the_inventory_json(self):
        plain = device("a")
        self.assertNotIn("kernels", device_to_json(plain))
        self.assertIsNone(device_from_json(device_to_json(plain)).kernels)
        rich = replace(plain, kernels=frozenset({"ffn", "attn_proj:fused"}))
        self.assertEqual(device_to_json(rich)["kernels"], ["attn_proj:fused", "ffn"])
        self.assertEqual(device_from_json(device_to_json(rich)).kernels, rich.kernels)
        with self.assertRaises(Exception):
            replace(plain, kernels=frozenset({"attn_proj:fused"}))      # ffn is always there


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.store = profile(a=priors(), b=priors(rate=20e9, overhead=4.0, transport="tcp"))
        self.problem = PlacementProblem(
            (model(layers=range(4)),),
            (device("a", capacity=10**12, stored_layers=MappingProxyType({"m": 0b11}),
                    qualified_layers=MappingProxyType({"m": 0b11})),
             device("b", formats={"m": "pk"}, transport="tcp", capacity=10**12)),
            self.store, latency_ppm=1_250_000)
        self.inputs = inputs(layers=5)

    def test_decision_table_names_gains_and_missing_kernels(self):
        table = og.decision_table(self.problem, self.inputs, tokens_per_run={"m": 1000.0})
        verdicts = table["verdicts"]["m"]
        self.assertEqual(verdicts["attn_proj fused"]["missing_kernel"], "attn_proj:fused")
        self.assertEqual(verdicts["output_head"]["missing_kernel"], "output_head")
        self.assertGreater(verdicts["attn_proj fused"]["gain_j_per_token"], 0)
        self.assertEqual(verdicts["attn_proj fused"]["verdict"], "offload")
        # the executable plan uses only stored + qualified layers and today's kernels
        executable = table["executable"]["models"]["m"]
        self.assertTrue(all(row["variant"] == "ffn" for row in executable["layers"]))
        self.assertTrue(all(row["ffn"] in (DESKTOP_CPU, "a") or row["layer"] < 2 for row in executable["layers"]))

    def test_output_head_respects_capacity_latency_and_max_models(self):
        store = profile(a=priors(rate=200e9, overhead=0.1, active_w=1.0))
        problem = PlacementProblem((model("m", layers=range(2)), model("n", layers=range(2))),
                                   (device("a", formats={"m": "f16", "n": "f16"}, max_models=1,
                                           kernels=frozenset({"ffn", "output_head"})),),
                                   store)
        opg = inputs(models=("m", "n"), layers=3)
        plan = og.solve_op_groups(problem, opg)
        served = {model_id for model_id, row in plan.models.items()
                  if any(layer["ffn"] == "a" for layer in row["layers"]) or row["output_head"]["owner"] == "a"}
        self.assertLessEqual(len(served), 1)
        for model_id, row in plan.models.items():
            for option in row["output_head"]["options"]:
                if option["owner"] == "a" and model_id not in served:
                    self.assertFalse(option["feasible"])

    def test_context_sweep_raises_host_kv_time_with_context(self):
        rows = og.context_sweep(self.problem, self.inputs, (1000, 64000), tokens_per_run={"m": 1000.0})
        self.assertLess(rows[0]["models"]["m"]["host_kv_ms_per_token_rows1"],
                        rows[1]["models"]["m"]["host_kv_ms_per_token_rows1"])

    def test_scaled_profile_moves_one_input(self):
        moved = og.scaled_profile(self.store, desktop_wait=1.25)
        self.assertAlmostEqual(moved.desktop_power.wait_w, 62.5)
        faster = og.scaled_profile(self.store, phone_bandwidth=1.25)
        self.assertAlmostEqual(faster.device_priors["a"].bytes_per_s["f16"], 62.5e9)
        rows = og.sensitivity(self.problem, self.inputs, tokens_per_run={"m": 1000.0}, factors=(1.25,))
        self.assertEqual({row["parameter"] for row in rows}, {"phone_energy", "desktop_wait", "phone_bandwidth"})


if __name__ == "__main__":
    unittest.main()
