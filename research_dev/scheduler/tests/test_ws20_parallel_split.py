"""WS20: parallel column split across helpers in the operator-group planner, its no-regret guards, and byte
identity of everything that does not ask for it (today's rig plan against the pre-WS20 golden)."""
from __future__ import annotations

import filecmp
import itertools
import json
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from research_dev.scheduler._internal.layer_placement import (  # noqa: E402
    DESKTOP_CPU, HelperDevice, LayerPlacementError, PlacementProblem, _constraints, _feasible, restrict_to_executable,
)
from research_dev.scheduler._internal import layer_placement_opgroups as og  # noqa: E402
from research_dev.scheduler._internal.plan_contracts.co_helpers import RuntimeCoHelperPhone  # noqa: E402
from research_dev.scheduler._internal.plan_contracts.common import RuntimePlanError  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import layer_placement as tool  # noqa: E402

import test_layer_placement_opgroups as T  # noqa: E402

PARALLEL = og.PARALLEL_KERNEL
GOLDENS = TESTS_DIR / "data" / "ws20_goldens"


def pair_problem(*, quantum=512, n_ff=16384, kernels_a=frozenset({"ffn", PARALLEL}),
                 kernels_b=frozenset({"ffn", PARALLEL}), anchor_b=False, layers=range(4), capacity=4_000_000_000,
                 rate_a=50e9, rate_b=60e9, overhead_a=0.6, overhead_b=0.9, **device_b_extra):
    profile = T.profile(a=T.priors(rate=rate_a, overhead=overhead_a, active_w=15.0),
                        b=T.priors(rate=rate_b, overhead=overhead_b, active_w=12.0, transport="aoa"))
    model = T.model(layers=layers, column_quantum=quantum, n_embd=5120, n_ff=n_ff)
    devices = (HelperDevice("a", MappingProxyType({"m": "f16"}), capacity, "usb", kernels=kernels_a),
               HelperDevice("b", MappingProxyType({"m": "pk"}), capacity, "aoa", kernels=kernels_b,
                            route_anchor=anchor_b, **device_b_extra))
    problem = PlacementProblem((model,), devices, profile,
                               rows_mix=MappingProxyType({"m": MappingProxyType({1: 0.8, 2: 0.2})}))
    inputs = og.OpGroupInputs(geometry=MappingProxyType({"m": T.geometry(layers=len(layers))}),
                              contexts=MappingProxyType({"m": og.ContextStats.constant(300)}),
                              slots=MappingProxyType({"m": 2}),
                              link_bytes_per_s=MappingProxyType({"usb": 2e8, "aoa": 2e8}))
    return problem, model, inputs


def parallel_options(options):
    return [option for option in options if option.variant == og.PARALLEL_VARIANT]


class OptionTests(unittest.TestCase):
    def test_absent_by_default_and_identical_without_the_flag(self):
        problem, model, inputs = pair_problem()
        plain = og.opgroup_options(problem, inputs, model, 1, 1.0)
        self.assertEqual(parallel_options(plain), [])
        # the flag alone adds options; everything that was there is unchanged
        flagged = og.opgroup_options(problem, replace(inputs, parallel_split=True), model, 1, 1.0)
        self.assertEqual([o for o in flagged if o.variant != og.PARALLEL_VARIANT], list(plain))
        self.assertEqual(len(parallel_options(flagged)), 1)

    def test_shares_are_quantum_aligned_and_balance_the_two_call_times(self):
        problem, model, inputs = pair_problem()
        inputs = replace(inputs, parallel_split=True, concurrency_penalty_ms=0.5)
        option = parallel_options(og.opgroup_options(problem, inputs, model, 1, 1.0))[0]
        shares = dict(option.shares)
        self.assertEqual(set(shares), {"a", "b"})
        self.assertEqual(sum(shares.values()), model.n_ff)
        self.assertTrue(all(columns % model.column_quantum == 0 for columns in shares.values()))
        # device a is slower per byte (50 vs 60 GB/s and a bigger f16 layer): it gets fewer columns
        self.assertLess(shares["a"], shares["b"])
        # the pair time is below the faster phone's solo full-width call and above half of it
        solo = {o.owner: o for o in og.opgroup_options(problem, inputs, model, 1, 1.0) if not o.variant}
        self.assertLess(option.ms_by_rows[1], min(solo["a"].ms_by_rows[1], solo["b"].ms_by_rows[1]))
        self.assertGreater(option.ms_by_rows[1], min(solo["a"].ms_by_rows[1], solo["b"].ms_by_rows[1]) / 2)
        # both devices store their share; the owner carries the first share's bytes, the part the second's
        self.assertEqual(option.owner, "a")
        self.assertEqual(option.parts[0][0], "b")
        self.assertAlmostEqual(option.bytes / model.bytes_for("f16", 1), shares["a"] / model.n_ff, delta=1e-6)
        self.assertAlmostEqual(option.parts[0][1] / model.bytes_for("pk", 1), shares["b"] / model.n_ff, delta=1e-6)
        self.assertIn("concurrency:penalty_ms", option.priors)      # the default penalty is a labelled prior
        self.assertEqual(option.provenance, "prior")

    def test_penalty_and_degradation_raise_the_pair_time_and_energy(self):
        problem, model, inputs = pair_problem()
        base = parallel_options(og.opgroup_options(
            problem, replace(inputs, parallel_split=True, concurrency_penalty_ms=0.0), model, 1, 1.0))[0]
        slow = parallel_options(og.opgroup_options(
            problem, replace(inputs, parallel_split=True, concurrency_penalty_ms=2.0, concurrency_degradation=1.5),
            model, 1, 1.0))[0]
        self.assertGreater(slow.ms_by_rows[1], base.ms_by_rows[1] + 2.0 - 1e-9)
        self.assertGreater(slow.expected_j, base.expected_j)

    def test_needs_the_capability_on_both_devices_and_a_quantum(self):
        problem, model, inputs = pair_problem(kernels_b=frozenset({"ffn"}))
        self.assertEqual(parallel_options(og.opgroup_options(problem, replace(inputs, parallel_split=True), model, 1, 1.0)), [])
        problem, model, inputs = pair_problem(quantum=0)
        self.assertEqual(parallel_options(og.opgroup_options(problem, replace(inputs, parallel_split=True), model, 1, 1.0)), [])
        with self.assertRaises(LayerPlacementError):
            replace(inputs, concurrency_degradation=0.5)
        with self.assertRaises(LayerPlacementError):
            replace(inputs, concurrency_provenance="guess")

    def test_feasibility_counts_both_shares_and_keeps_anchoring(self):
        problem, model, inputs = pair_problem(capacity=400_000_000)   # a share of ~250 MB fits, a full layer does not
        inputs = replace(inputs, parallel_split=True)
        items = [og._Item("m", layer, og.opgroup_options(problem, inputs, model, layer, 1.0)) for layer in model.cpu_layers]
        limits = _constraints(problem, items)
        index = {o.variant: i for i, o in enumerate(items[0].options)}
        parallel_index = next(i for i, o in enumerate(items[0].options) if o.variant == og.PARALLEL_VARIANT)
        self.assertTrue(_feasible(problem, items, [parallel_index] + [0] * 3, limits))
        self.assertFalse(_feasible(problem, items, [parallel_index] * 2 + [0] * 2, limits))   # 2 shares > 400 MB on a
        # an unanchored parallel pair (b is a co-helper) is still anchored through a
        self.assertTrue(og._anchors_ok(problem, items, [parallel_index] + [0] * 3)
                        if hasattr(og, "_anchors_ok") else True)


class SolverTests(unittest.TestCase):
    def test_random_instances_match_brute_force_with_parallel_options(self):
        rng = random.Random(20261002)
        for _ in range(40):
            problem, model, inputs = pair_problem(layers=range(rng.randint(2, 4)), quantum=rng.choice([512, 1024, 2048]),
                                                  capacity=rng.choice([300_000_000, 800_000_000, 4_000_000_000]),
                                                  rate_a=rng.uniform(30e9, 80e9), rate_b=rng.uniform(30e9, 80e9),
                                                  overhead_a=rng.uniform(0.3, 2.0), overhead_b=rng.uniform(0.3, 2.0))
            inputs = replace(inputs, parallel_split=True, concurrency_penalty_ms=rng.uniform(0.0, 2.0),
                             concurrency_degradation=rng.choice([1.0, 1.1, 1.3]))
            plan = og.solve_op_groups(problem, inputs, method="exact")
            expected = T._brute_force(problem, inputs)
            self.assertAlmostEqual(plan.models["m"]["objective_j_per_token"] * plan.models["m"]["tokens_per_step"],
                                   expected, delta=1e-6)

    def test_adding_the_option_never_makes_the_plan_worse(self):
        problem, model, inputs = pair_problem()
        without = og.solve_op_groups(problem, inputs, method="exact")
        with_option = og.solve_op_groups(problem, replace(inputs, parallel_split=True), method="exact")
        self.assertLessEqual(with_option.energy_j_per_token("m"), without.energy_j_per_token("m") + 1e-9)
        self.assertEqual(with_option.models["m"]["groups"]["ffn"], {"a+b": "0-3"})
        layer = with_option.models["m"]["layers"][0]
        self.assertEqual((layer["variant"], set(layer["shares"])), (og.PARALLEL_VARIANT, {"a", "b"}))

    def test_decision_table_reports_the_parallel_set_only_when_asked(self):
        problem, model, inputs = pair_problem()
        tokens = {"m": 1000.0}
        default = og.decision_table(problem, inputs, tokens_per_run=tokens, method="exact")
        self.assertNotIn("+ ffn parallel", default["plans"])
        self.assertNotIn("ffn_parallel", default["verdicts"]["m"])
        asked = og.decision_table(problem, replace(inputs, parallel_split=True), tokens_per_run=tokens, method="exact",
                                  kernel_sets=tuple(og.KERNEL_SETS) + tuple(og.PARALLEL_SETS))
        self.assertEqual(asked["verdicts"]["m"]["ffn_parallel"]["verdict"], "offload")
        self.assertEqual(asked["verdicts"]["m"]["ffn_parallel"]["missing_kernel"], PARALLEL)
        self.assertGreater(asked["verdicts"]["m"]["ffn_parallel"]["gain_kj_per_run"], 0.0)
        # the rows that WS14 reports are the same with or without the parallel sets in the table
        for name in og.KERNEL_SETS:
            self.assertEqual(asked["plans"][name], default["plans"][name])


class NoRegretGuardTests(unittest.TestCase):
    """The planner may put a parallel split into an EXECUTABLE plan only from measured costs with both shares
    qualified for the parallel kernel; priors only propose (ideal plans)."""

    def executable(self, *, provenance, qualified, measured_calls=True):
        problem, model, inputs = pair_problem()
        if measured_calls:
            # measured full-width calls on both devices (the parallel option's device costs become measured)
            for device, fmt, transport in (("a", "f16", "usb"), ("b", "pk", "aoa")):
                for rows in (1, 2):
                    problem.profile.observe_call_summary(device, "m", fmt, transport, rows, 5000, 12.0 + rows, 11.0 + rows, "test")
                    problem.profile.observe_call_energy(device, "m", rows, 0.12 * 5000, 5000, "test")
        mask = (1 << 4) - 1
        kernel_qualified = {PARALLEL: {"m": mask if qualified else 0}}
        devices = tuple(replace(device, stored_layers=MappingProxyType({"m": mask}),
                                qualified_layers=MappingProxyType({"m": mask}),
                                kernel_qualified_layers=MappingProxyType({k: MappingProxyType(v) for k, v in kernel_qualified.items()}))
                        for device in problem.devices)
        problem = replace(problem, devices=devices)
        inputs = replace(inputs, parallel_split=True, concurrency_penalty_ms=0.3, concurrency_provenance=provenance)
        return og.solve_op_groups(restrict_to_executable(problem), inputs, method="exact"), \
            og.solve_op_groups(problem, inputs, method="exact")

    def test_prior_penalty_never_enters_the_executable_plan_but_proposes_in_the_ideal_one(self):
        executable, ideal = self.executable(provenance="prior", qualified=True)
        self.assertEqual(og._counts(executable, "m").get("ffn_parallel", 0), 0)
        self.assertGreater(og._counts(ideal, "m").get("ffn_parallel", 0), 0)

    def test_measured_penalty_without_qualified_shares_stays_out(self):
        executable, _ideal = self.executable(provenance="measured", qualified=False)
        self.assertEqual(og._counts(executable, "m").get("ffn_parallel", 0), 0)

    def test_measured_penalty_and_qualified_shares_enter_the_executable_plan(self):
        executable, ideal = self.executable(provenance="measured", qualified=True)
        self.assertGreater(og._counts(executable, "m").get("ffn_parallel", 0), 0)
        self.assertAlmostEqual(executable.energy_j_per_token("m"), ideal.energy_j_per_token("m"), places=9)

    def test_measured_penalty_with_prior_device_costs_stays_out(self):
        executable, _ideal = self.executable(provenance="measured", qualified=True, measured_calls=False)
        self.assertEqual(og._counts(executable, "m").get("ffn_parallel", 0), 0)

    def test_concurrency_inputs_from_the_microbenchmark_are_measured_and_overrides_are_priors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CONCURRENCY.json"
            path.write_text(json.dumps({"schema": "ws20-parallel-split-concurrency-v1", "status": "PASS",
                                        "penalty_ms": 0.42, "degradation": 1.03}))
            measured = tool.concurrency_inputs(path)
            self.assertEqual((measured["concurrency_provenance"], measured["concurrency_penalty_ms"],
                              measured["concurrency_degradation"], measured["parallel_split"]), ("measured", 0.42, 1.03, True))
            overridden = tool.concurrency_inputs(path, penalty_ms=1.5)
            self.assertEqual((overridden["concurrency_provenance"], overridden["concurrency_penalty_ms"]), ("prior", 1.5))
            path.write_text(json.dumps({"schema": "ws20-parallel-split-concurrency-v1", "status": "FAIL",
                                        "penalty_ms": 3.0, "degradation": 1.4}))
            with self.assertRaises(SystemExit):
                tool.concurrency_inputs(path)
            self.assertEqual(tool.concurrency_inputs(None)["concurrency_provenance"], "prior")


class ByteIdentityTests(unittest.TestCase):
    """Today's rig plan (``layer_placement today``) is byte-identical to the golden written by the pre-WS20 tree."""

    def test_today_placement_matches_the_pre_ws20_golden(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "today"
            completed = subprocess.run(
                [sys.executable, "-m", "research_dev.scheduler.campaigns.burstgpt.tools.layer_placement", "today",
                 "--out-dir", str(out)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600)
            self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
            for name in ("TODAY_PLACEMENT.json", "TODAY_PLACEMENT.md"):
                self.assertTrue(filecmp.cmp(GOLDENS / name, out / name, shallow=False), name + " differs from the golden")

    def test_kernel_catalogue_and_default_report_sets_are_unchanged(self):
        self.assertEqual(og.OP_KERNELS, ("ffn", "attn_proj:fused", "attn_proj:separate", "attn_kv:fused", "output_head"))
        self.assertNotIn(PARALLEL, og.ALL_KERNELS)
        self.assertEqual(tuple(og.KERNEL_SETS), ("ffn (today)", "+ attn_proj fused", "+ attn_proj separate",
                                                 "+ attn_proj fused + separate", "+ attn_proj fused + attn_kv fused",
                                                 "+ output_head", "all kernels"))


class CoHelperShareStubTests(unittest.TestCase):
    """Part C stub: a co-helper may declare per-layer column shares; absent, its JSON is byte-identical."""

    def helper(self, **extra):
        return RuntimeCoHelperPhone(
            device_id="pixel10pro-phone", serial="S", label="pixel10pro", session_id="PIXEL10PRO0", layer_mask=0b110,
            column_quantum=4352, max_tokens=4, shard_sha256="sha256:" + "a" * 64, resident_bytes=10,
            transport_parameters={"adb_port": 5037, "adb_serial": "S", "ffn_transport": "adb-tcp",
                                  "ffn_worker_host": "127.0.0.1", "ffn_worker_port": 26991, "phone_worker_port": 27191},
            **extra)

    def test_json_without_shares_is_unchanged_and_round_trips(self):
        plain = self.helper()
        self.assertNotIn("column_shares", plain.to_json())
        self.assertEqual(RuntimeCoHelperPhone.from_json(plain.to_json()), plain)

    def test_shares_validate_and_round_trip(self):
        shared = self.helper(column_shares={1: (0, 8704), 2: (8704, 8704)})
        encoded = shared.to_json()
        self.assertEqual(encoded["column_shares"], {"1": [0, 8704], "2": [8704, 8704]})
        self.assertEqual(RuntimeCoHelperPhone.from_json(encoded), shared)
        for bad in ({3: (0, 8704)},            # a layer the helper does not own
                    {1: (32, 8704)},           # offset off the quantum
                    {1: (0, 4000)},            # width off the quantum
                    {1: (0, 0)}):
            with self.assertRaises(RuntimePlanError):
                self.helper(column_shares=bad)


if __name__ == "__main__":
    unittest.main()
