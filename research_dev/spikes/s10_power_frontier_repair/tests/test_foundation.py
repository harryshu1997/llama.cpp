#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "oracle"), str(ROOT / "checker"),
                str(ROOT / "policies"), str(ROOT / "tests")]
import exact  # noqa: E402
import checker  # noqa: E402
import causal  # noqa: E402
import slow_reference  # noqa: E402
import gen_cases  # noqa: E402
import differential  # noqa: E402


def route(duration, extra=0):
    return {"duration_us": duration, "extra_energy_nj": extra}


def base_power():
    return {"p8_mw": 25000, "p0_mw": 300000, "wake_us": 0,
            "idle_entry_us": 0, "transition_nj": 0}


def partial_partition_instance():
    deadlines = [550, 1200, 1200]
    requests = []
    nodes = []
    for index, deadline in enumerate(deadlines):
        rid = f"r{index}"
        nid = f"n{index}"
        requests.append({"id": rid, "arrival_us": 0, "terminal_node": nid,
                         "deadline_us": deadline, "priority": 0})
        nodes.append({
            "id": nid, "request_id": rid, "model_id": "m0",
            "weight_set_id": "w0", "predecessors": [], "release_us": 0,
            "routes": {"SERVER": route(545)}, "batch_key": "ffn:w0",
            "tokens": 16, "output_bytes": 0,
        })
    return {
        "schema_version": 2,
        "instance_id": "partial_partition",
        "horizon_us": 5000,
        "activation_mem_bound_bytes": 1024,
        "server_power": base_power(),
        "devices": {"SERVER": {"kind": "server", "active_mw": 300000}},
        "batch_profiles": {"ffn:w0": {"16": 545, "32": 599, "48": 607}},
        "requests": requests,
        "nodes": nodes,
        "evidence": {"scope": "MECHANICS_ONLY",
                     "latency": "retained V0 measured/derived table"},
    }


def causal_pair():
    current_request = {"id": "current", "arrival_us": 0, "terminal_node": "current_mid",
                       "deadline_us": 40000, "priority": 0}
    current_node = {
        "id": "current_mid", "request_id": "current", "model_id": "m0",
        "weight_set_id": "w0", "predecessors": [], "release_us": 0,
        "routes": {"SERVER": route(545), "OP15": route(27142)},
        "batch_key": "ffn:w0", "tokens": 16, "output_bytes": 0,
    }
    base = {
        "schema_version": 2,
        "instance_id": "causal_a",
        "horizon_us": 100000,
        "activation_mem_bound_bytes": 1024,
        "server_power": base_power(),
        "devices": {
            "SERVER": {"kind": "server", "active_mw": 300000},
            "OP15": {"kind": "phone", "active_mw": 2000},
        },
        "batch_profiles": {"ffn:w0": {"16": 545, "32": 599}},
        "requests": [current_request],
        "nodes": [current_node],
        "evidence": {"scope": "MECHANICS_ONLY",
                     "power": "inferred favorable for regression only"},
    }
    a = copy.deepcopy(base)
    b = copy.deepcopy(base)
    b["instance_id"] = "causal_b"
    b["requests"].append({"id": "future", "arrival_us": 10000,
                          "terminal_node": "future_mid", "deadline_us": 30000,
                          "priority": 0})
    b["nodes"].append({
        "id": "future_mid", "request_id": "future", "model_id": "m1",
        "weight_set_id": "w1", "predecessors": [], "release_us": 10000,
        "routes": {"SERVER": route(545)}, "batch_key": "ffn:w1",
        "tokens": 16, "output_bytes": 0,
    })
    b["batch_profiles"]["ffn:w1"] = {"16": 545}
    return a, b


def activation_instance(bound):
    requests = [
        {"id": "r0", "arrival_us": 0, "terminal_node": "m0", "deadline_us": 500,
         "priority": 0},
        {"id": "r1", "arrival_us": 0, "terminal_node": "m1", "deadline_us": 500,
         "priority": 0},
    ]
    nodes = [
        {"id": "p0", "request_id": "r0", "model_id": "m", "weight_set_id": "wp",
         "predecessors": [], "release_us": 0, "routes": {"SERVER": route(10)},
         "batch_key": None, "tokens": 1, "output_bytes": 100},
        {"id": "m0", "request_id": "r0", "model_id": "m", "weight_set_id": "w0",
         "predecessors": ["p0"], "release_us": 0, "routes": {"OP15": route(100)},
         "batch_key": None, "tokens": 1, "output_bytes": 0},
        {"id": "p1", "request_id": "r1", "model_id": "m", "weight_set_id": "wp",
         "predecessors": [], "release_us": 0, "routes": {"SERVER": route(10)},
         "batch_key": None, "tokens": 1, "output_bytes": 100},
        {"id": "m1", "request_id": "r1", "model_id": "m", "weight_set_id": "w1",
         "predecessors": ["p1"], "release_us": 0, "routes": {"OP15": route(100)},
         "batch_key": None, "tokens": 1, "output_bytes": 0},
    ]
    return {
        "schema_version": 2, "instance_id": "activation", "horizon_us": 1000,
        "activation_mem_bound_bytes": bound, "server_power": base_power(),
        "devices": {"SERVER": {"kind": "server", "active_mw": 300000},
                    "OP15": {"kind": "phone", "active_mw": 2000}},
        "batch_profiles": {}, "requests": requests, "nodes": nodes,
        "evidence": {"scope": "MECHANICS_ONLY",
                     "purpose": "activation lifetime regression"},
    }


def certificate_for_actions(inst, actions):
    reqs, nodes = exact.validate_instance(inst)
    result = exact.evaluate(inst, reqs, nodes, actions)
    if result is None:
        raise ValueError("actions are infeasible under instance bound")
    cert = {
        "schema_version": 2,
        "instance_id": inst["instance_id"],
        "instance_sha256": exact.digest(inst),
        "actions": actions,
        "request_outcomes": result["request_outcomes"],
        "activation_peak_bytes": result["activation_peak_bytes"],
        "energy": result["energy"],
        "objective": result["objective"],
        "search": {"complete": False},
    }
    cert["certificate_sha256"] = exact.digest(cert)
    return cert


def reseal(cert):
    body = {key: value for key, value in cert.items() if key != "certificate_sha256"}
    cert["certificate_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def rebind(inst, cert):
    cert["instance_sha256"] = exact.digest(inst)
    reseal(cert)


class FoundationTests(unittest.TestCase):
    def test_differential_rejects_paired_unexpected_errors(self):
        """Two arbitrary RuntimeErrors are not evidence of joint infeasibility."""
        inst = partial_partition_instance()
        with mock.patch.object(differential.gen_cases, "make_case", return_value=inst), \
                mock.patch.object(
                    differential.exact, "solve",
                    side_effect=RuntimeError(
                        "exact state bound exceeded; no approximate result emitted")), \
                mock.patch.object(
                    differential.reference, "optimum",
                    side_effect=RuntimeError("unexpected reference failure")):
            result = differential.run(0, 1)
        self.assertEqual(result["agreed_infeasible"], 0)
        self.assertEqual(len(result["mismatches"]), 1)
        self.assertEqual(result["mismatches"][0]["oracle_status"], "error")
        self.assertEqual(result["mismatches"][0]["reference_status"], "error")

    def test_all_set_partitions_present(self):
        self.assertEqual(len(list(exact.set_partitions(["a", "b", "c", "d"]))), 15)

    def test_partial_partition_counterexample(self):
        inst = partial_partition_instance()
        cert = exact.solve(inst)
        self.assertEqual(cert["objective"][0], 0)
        member_sets = {tuple(sorted(action["members"])) for action in cert["actions"]}
        self.assertIn(("n0",), member_sets)
        self.assertIn(("n1", "n2"), member_sets)
        self.assertEqual(checker.check(inst, cert, require_complete=False), [])
        # This fixture's horizon is far outside the independent reference's tiny
        # declared domain, so default mode certifies NOTHING and says so. It must
        # never fall back to trusting the certificate's own search counters.
        failures = checker.check(inst, cert)
        self.assertTrue(any("not independently verifiable" in failure
                            for failure in failures), failures)

    def test_exact_matches_independent_reference_generated(self):
        rng = random.Random(89458)
        for case in range(50):
            requests = []
            nodes = []
            for request_index in range(2):
                previous = None
                chain_length = 1 + rng.randrange(2)
                for node_index in range(chain_length):
                    nid = f"n{request_index}_{node_index}"
                    routes = {"SERVER": route(rng.randint(1, 5))}
                    if rng.random() < 0.7:
                        routes["PHONE"] = route(rng.randint(2, 8), rng.randint(0, 5))
                    nodes.append({
                        "id": nid,
                        "request_id": f"r{request_index}",
                        "model_id": "m0",
                        "weight_set_id": f"w{node_index}",
                        "predecessors": [] if previous is None else [previous],
                        "release_us": rng.randint(0, 3),
                        "routes": routes,
                        "batch_key": None,
                        "tokens": 1,
                        "output_bytes": 0,
                    })
                    previous = nid
                requests.append({
                    "id": f"r{request_index}",
                    "arrival_us": 0,
                    "terminal_node": previous,
                    "deadline_us": rng.randint(5, 20),
                    "priority": 0,
                })
            inst = {
                "schema_version": 2,
                "instance_id": f"generated_{case}",
                "horizon_us": 25,
                "activation_mem_bound_bytes": 100,
                "server_power": {
                    "p8_mw": 1, "p0_mw": 10, "wake_us": 0,
                    "idle_entry_us": 0, "transition_nj": 0,
                },
                "devices": {
                    "SERVER": {"kind": "server", "active_mw": 10},
                    "PHONE": {"kind": "phone", "active_mw": rng.randint(1, 5)},
                },
                "batch_profiles": {},
                "requests": requests,
                "nodes": nodes,
                "evidence": {"scope": "MECHANICS_ONLY",
                             "purpose": "independent generated reference"},
            }
            expected = slow_reference.solve_objective(inst)
            actual = tuple(exact.solve(inst, max_states=200000)["objective"])
            self.assertEqual(actual, expected, msg=f"generated case {case}")

    def test_batch_partitions_match_independent_reference_generated(self):
        rng = random.Random(10510)
        for case in range(40):
            count = rng.randint(2, 5)
            profile = {str(tokens): rng.randint(2, 7)
                       for tokens in range(1, count + 1)}
            requests = []
            nodes = []
            for index in range(count):
                rid = f"r{index}"
                nid = f"n{index}"
                requests.append({
                    "id": rid,
                    "arrival_us": 0,
                    "terminal_node": nid,
                    "deadline_us": rng.randint(3, 25),
                    "priority": 0,
                })
                nodes.append({
                    "id": nid,
                    "request_id": rid,
                    "model_id": "m0",
                    "weight_set_id": "w0",
                    "predecessors": [],
                    "release_us": rng.randint(0, 4),
                    "routes": {"SERVER": route(profile["1"])},
                    "batch_key": "op:w0",
                    "tokens": 1,
                    "output_bytes": 0,
                })
            inst = {
                "schema_version": 2,
                "instance_id": f"batch_generated_{case}",
                "horizon_us": 40,
                "activation_mem_bound_bytes": 100,
                "server_power": {
                    "p8_mw": 1,
                    "p0_mw": 10,
                    "wake_us": 0,
                    "idle_entry_us": 0,
                    "transition_nj": 0,
                },
                "devices": {"SERVER": {"kind": "server", "active_mw": 10}},
                "batch_profiles": {"op:w0": profile},
                "requests": requests,
                "nodes": nodes,
                "evidence": {
                    "scope": "MECHANICS_ONLY",
                    "purpose": "independent batch-partition reference",
                },
            }
            expected = slow_reference.solve_single_server_batch_objective(inst)
            actual = tuple(exact.solve(inst, max_states=200000)["objective"])
            self.assertEqual(actual, expected, msg=f"batched generated case {case}")

    def test_causal_prefix_invariance(self):
        left, right = causal_pair()
        self.assertEqual(causal.first_action(left, 0), causal.first_action(right, 0))
        self.assertEqual(causal.first_action(left, 5000), causal.first_action(right, 5000))

    def test_causal_snapshot_never_returns_past_action(self):
        inst, _ = causal_pair()
        action = causal.first_action(inst, 5000)
        self.assertGreaterEqual(action["planned_start_us"], 5000)
        with self.assertRaises(ValueError):
            causal.first_action(inst, -1)

    def test_transition_energy_depends_on_gap_placement(self):
        inst = partial_partition_instance()
        inst["horizon_us"] = 2000
        inst["server_power"] = {"p8_mw": 25000, "p0_mw": 300000,
                                "wake_us": 50, "idle_entry_us": 50,
                                "transition_nj": 1000000}
        clustered = [
            {"id": "a0", "device": "SERVER", "members": ["n0"],
             "start_us": 100, "finish_us": 200},
            {"id": "a1", "device": "SERVER", "members": ["n1"],
             "start_us": 220, "finish_us": 320},
        ]
        separated = copy.deepcopy(clustered)
        separated[1]["start_us"] = 1000
        separated[1]["finish_us"] = 1100
        oracle_clustered = exact._server_energy(inst, clustered)
        oracle_separated = exact._server_energy(inst, separated)
        checker_clustered = checker._server_energy(inst, clustered)
        checker_separated = checker._server_energy(inst, separated)
        self.assertEqual(oracle_clustered, checker_clustered)
        self.assertEqual(oracle_separated, checker_separated)
        self.assertNotEqual(oracle_clustered[0], oracle_separated[0])
        self.assertEqual(sum(a["finish_us"] - a["start_us"] for a in clustered),
                         sum(a["finish_us"] - a["start_us"] for a in separated))

    def test_transition_delay_counterexample_is_selected(self):
        path = ROOT / "fixtures" / "transition_delay_counterexample.json"
        inst = json.loads(path.read_text(encoding="ascii"))
        left_shifted = [
            {"id": "a0", "device": "SERVER", "members": ["n0"],
             "start_us": 50, "finish_us": 150},
            {"id": "a1", "device": "SERVER", "members": ["n1"],
             "start_us": 1000, "finish_us": 1100},
        ]
        delayed = [
            {"id": "a0", "device": "SERVER", "members": ["n0"],
             "start_us": 850, "finish_us": 950},
            {"id": "a1", "device": "SERVER", "members": ["n1"],
             "start_us": 1000, "finish_us": 1100},
        ]
        left_cert = certificate_for_actions(inst, left_shifted)
        delayed_cert = certificate_for_actions(inst, delayed)
        self.assertEqual(checker.check(inst, left_cert),
                         ["search.complete must be true"])
        self.assertEqual(checker.check(inst, delayed_cert),
                         ["search.complete must be true"])
        self.assertEqual(checker.check(inst, left_cert, require_complete=False), [])
        self.assertEqual(checker.check(inst, delayed_cert, require_complete=False), [])
        self.assertLess(delayed_cert["energy"]["total_nj"],
                        left_cert["energy"]["total_nj"])
        self.assertEqual(left_cert["energy"]["total_nj"], 162000000)
        self.assertEqual(delayed_cert["energy"]["total_nj"], 147250000)

        # The exact solver must now enumerate intentional delay and SELECT the
        # lower-energy delayed placement at identical zero-miss/zero-lateness
        # outcomes, instead of refusing or returning the earliest schedule.
        cert = exact.solve(inst)
        self.assertTrue(cert["search"]["complete"])
        self.assertEqual(cert["energy"]["total_nj"], 147250000)
        self.assertEqual(cert["objective"], [0, 0, -2, 147250000])
        placement = {tuple(action["members"]): (action["start_us"], action["finish_us"])
                     for action in cert["actions"]}
        self.assertEqual(placement[("n0",)], (850, 950))
        self.assertEqual(placement[("n1",)], (1000, 1100))
        # one merged P0 window, not two
        self.assertEqual(cert["energy"]["server_p0_intervals"], [[800, 1150]])
        self.assertEqual(checker.check(inst, cert, require_complete=False), [])

    def test_pruning_agrees_with_unpruned_search(self):
        """Documented B&B rules R1/R2 must not change the optimum (CP1 contract).

        Covers the frozen activation fixture plus generated temporal cases; the
        independent reference (which uses no incumbent/objective pruning) covers
        the frozen transition fixture and the generated corpus separately.
        """
        path = ROOT / "fixtures" / "activation_delay_counterexample.json"
        cases = [json.loads(path.read_text(encoding="ascii"))]
        cases.extend(gen_cases.make_case(seed) for seed in (1, 2, 3, 5, 7, 11))
        compared = 0
        for inst in cases:
            outcomes = []
            for prune in (True, False):
                try:
                    cert = exact.solve(inst, prune=prune)
                except RuntimeError as exc:
                    self.assertEqual(str(exc), "no feasible complete schedule",
                                     msg=inst["instance_id"])
                    outcomes.append(("infeasible",))
                else:
                    outcomes.append(("feasible", cert["objective"], cert["actions"]))
            self.assertEqual(outcomes[0], outcomes[1], msg=inst["instance_id"])
            compared += 1
        self.assertEqual(compared, len(cases))

    def test_partial_partition_optimum_matches_independent_batch_reference(self):
        """The EARLIEST-mode frozen fixture is independently verified too.

        Its horizon puts it outside checker/reference.py's declared tiny domain, so
        default checker mode fails closed on it. The separately written batch
        reference in tests/slow_reference.py still proves its optimum, so no frozen
        fixture rests on the oracle's own word.
        """
        inst = partial_partition_instance()
        expected = slow_reference.solve_single_server_batch_objective(inst)
        actual = tuple(exact.solve(inst)["objective"])
        self.assertEqual(actual, expected)

    def test_activation_delay_counterexample_is_solved(self):
        """Earliest placement breaks the bound; a delayed placement is feasible."""
        path = ROOT / "fixtures" / "activation_delay_counterexample.json"
        inst = json.loads(path.read_text(encoding="ascii"))
        reqs, nodes = exact.validate_instance(inst)
        earliest = [
            {"id": "a00", "device": "SERVER", "members": ["p0"],
             "start_us": 0, "finish_us": 4},
            {"id": "a01", "device": "SERVER", "members": ["p1"],
             "start_us": 4, "finish_us": 8},
            {"id": "a02", "device": "OP15", "members": ["c0"],
             "start_us": 4, "finish_us": 10},
            {"id": "a03", "device": "OP15", "members": ["c1"],
             "start_us": 10, "finish_us": 16},
        ]
        self.assertEqual(exact._activation_peak(inst, nodes, earliest), 200)
        self.assertGreater(200, inst["activation_mem_bound_bytes"])
        self.assertIsNone(exact.evaluate(inst, reqs, nodes, earliest))

        cert = exact.solve(inst)
        self.assertTrue(cert["search"]["complete"])
        self.assertEqual(cert["objective"][0], 0)
        self.assertLessEqual(cert["activation_peak_bytes"],
                             inst["activation_mem_bound_bytes"])
        self.assertEqual(checker.check(inst, cert, require_complete=False), [])

    def test_temporal_domain_exhaustion_fails_closed(self):
        """Out-of-domain temporal instances raise; they never claim completeness."""
        with self.assertRaisesRegex(RuntimeError, "domain exceeds the declared bound"):
            exact.solve(activation_instance(1000))
        path = ROOT / "fixtures" / "transition_delay_counterexample.json"
        inst = json.loads(path.read_text(encoding="ascii"))
        with self.assertRaisesRegex(RuntimeError, "state bound exceeded"):
            exact.solve(inst, max_states=10)

    def test_checker_rejects_resealed_tampering(self):
        inst = partial_partition_instance()
        cert = exact.solve(inst)
        self.assertEqual(checker.check(inst, cert, require_complete=False), [])
        mutations = []

        wrong_member = copy.deepcopy(cert)
        wrong_member["actions"][0]["members"] = ["phantom"]
        reseal(wrong_member)
        mutations.append(wrong_member)

        forged_objective = copy.deepcopy(cert)
        forged_objective["objective"] = [0, 0, -3, 0]
        reseal(forged_objective)
        mutations.append(forged_objective)

        phantom_hbm = copy.deepcopy(cert)
        phantom_hbm["hbm"] = {"phantom": "exclusive"}
        reseal(phantom_hbm)
        mutations.append(phantom_hbm)

        extra_phone_list = copy.deepcopy(cert)
        extra_phone_list["phone_islands"] = ["phantom"]
        reseal(extra_phone_list)
        mutations.append(extra_phone_list)

        forged_energy = copy.deepcopy(cert)
        forged_energy["energy"]["total_nj"] += 1
        reseal(forged_energy)
        mutations.append(forged_energy)

        for mutated in mutations:
            self.assertTrue(checker.check(inst, mutated, require_complete=False))

    def test_activation_lifetime_includes_queue_wait(self):
        high_bound = activation_instance(1000)
        actions = [
            {"id": "a0", "device": "SERVER", "members": ["p0"],
             "start_us": 0, "finish_us": 10},
            {"id": "a1", "device": "SERVER", "members": ["p1"],
             "start_us": 10, "finish_us": 20},
            {"id": "a2", "device": "OP15", "members": ["m0"],
             "start_us": 30, "finish_us": 130},
            {"id": "a3", "device": "OP15", "members": ["m1"],
             "start_us": 130, "finish_us": 230},
        ]
        cert = certificate_for_actions(high_bound, actions)
        self.assertEqual(cert["activation_peak_bytes"], 200)
        self.assertEqual(checker.check(high_bound, cert, require_complete=False), [])

        low_bound = copy.deepcopy(high_bound)
        low_bound["activation_mem_bound_bytes"] = 150
        cert["instance_sha256"] = exact.digest(low_bound)
        reseal(cert)
        failures = checker.check(low_bound, cert, require_complete=False)
        self.assertTrue(any("activation peak" in failure for failure in failures))

    def test_activation_lifetime_covers_consumer_execution(self):
        inst = activation_instance(1000)
        actions = [
            {"id": "a0", "device": "SERVER", "members": ["p0"],
             "start_us": 0, "finish_us": 10},
            {"id": "a1", "device": "SERVER", "members": ["p1"],
             "start_us": 10, "finish_us": 20},
            {"id": "a2", "device": "OP15", "members": ["m0"],
             "start_us": 10, "finish_us": 110},
            {"id": "a3", "device": "OP15", "members": ["m1"],
             "start_us": 110, "finish_us": 210},
        ]
        cert = certificate_for_actions(inst, actions)
        self.assertEqual(cert["activation_peak_bytes"], 200)
        self.assertEqual(checker.check(inst, cert, require_complete=False), [])

        low_bound = copy.deepcopy(inst)
        low_bound["activation_mem_bound_bytes"] = 100
        rebind(low_bound, cert)
        self.assertTrue(any("activation peak" in failure
                            for failure in checker.check(
                                low_bound, cert, require_complete=False)))

    def test_checker_does_not_certify_suboptimal_schedule(self):
        """Feasible-but-suboptimal must be rejected; the proven optimum accepted.

        Uses the frozen transition fixture because it is inside the independent
        reference's declared tiny domain, so default mode can actually prove the
        optimum rather than fail closed.
        """
        path = ROOT / "fixtures" / "transition_delay_counterexample.json"
        inst = json.loads(path.read_text(encoding="ascii"))
        optimum = exact.solve(inst)
        self.assertEqual(optimum["energy"]["total_nj"], 147250000)

        # the earliest/left schedule: feasible, identical outcomes, MORE energy
        suboptimal = certificate_for_actions(inst, [
            {"id": "a0", "device": "SERVER", "members": ["n0"],
             "start_us": 50, "finish_us": 150},
            {"id": "a1", "device": "SERVER", "members": ["n1"],
             "start_us": 1000, "finish_us": 1100},
        ])
        self.assertEqual(suboptimal["energy"]["total_nj"], 162000000)
        self.assertEqual(checker.check(inst, suboptimal, require_complete=False), [])
        self.assertGreater(tuple(suboptimal["objective"]), tuple(optimum["objective"]))

        # Copying the optimum's plausible search counters must NOT buy a pass:
        # default mode proves optimality independently and rejects this schedule.
        suboptimal["search"] = copy.deepcopy(optimum["search"])
        reseal(suboptimal)
        failures = checker.check(inst, suboptimal)
        self.assertTrue(any("SUBOPTIMAL" in failure for failure in failures), failures)

        # ...while the genuine optimum is accepted by the same default mode.
        self.assertEqual(checker.check(inst, optimum), [])

    def test_duplicate_json_keys_fail(self):
        with tempfile.NamedTemporaryFile("w", encoding="ascii", delete=False) as handle:
            handle.write('{"schema_version":2,"schema_version":2}')
            path = handle.name
        try:
            with self.assertRaises(checker.DuplicateKeyError):
                checker.load_strict(path)
        finally:
            pathlib.Path(path).unlink()

    def test_oracle_duplicate_json_keys_fail(self):
        with tempfile.NamedTemporaryFile("w", encoding="ascii", delete=False) as handle:
            handle.write('{"schema_version":2,"schema_version":2}')
            path = handle.name
        try:
            with self.assertRaises(exact.DuplicateKeyError):
                exact.load_strict(path)
        finally:
            pathlib.Path(path).unlink()

    def test_nonfinite_json_constants_fail(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with tempfile.NamedTemporaryFile("w", encoding="ascii", delete=False) as handle:
                handle.write('{"value":' + token + "}")
                path = handle.name
            try:
                with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                    exact.load_strict(path)
                with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                    checker.load_strict(path)
            finally:
                pathlib.Path(path).unlink()

    def test_terminal_must_close_request_dag(self):
        inst = activation_instance(1000)
        inst["requests"][0]["terminal_node"] = "p0"
        with self.assertRaisesRegex(ValueError, "terminal node is not a sink"):
            exact.validate_instance(inst)

        valid = activation_instance(1000)
        actions = [
            {"id": "a0", "device": "SERVER", "members": ["p0"],
             "start_us": 0, "finish_us": 10},
            {"id": "a1", "device": "SERVER", "members": ["p1"],
             "start_us": 10, "finish_us": 20},
            {"id": "a2", "device": "OP15", "members": ["m0"],
             "start_us": 20, "finish_us": 120},
            {"id": "a3", "device": "OP15", "members": ["m1"],
             "start_us": 120, "finish_us": 220},
        ]
        cert = certificate_for_actions(valid, actions)
        rebind(inst, cert)
        self.assertTrue(any("terminal node is not a sink" in failure
                            for failure in checker.check(
                                inst, cert, require_complete=False)))

    def test_checker_rejects_semantic_holes(self):
        base = partial_partition_instance()
        cert = exact.solve(base)

        negative_priority = copy.deepcopy(base)
        negative_priority["requests"][0]["priority"] = -1
        negative_cert = copy.deepcopy(cert)
        rebind(negative_priority, negative_cert)
        self.assertTrue(any("priority" in failure
                            for failure in checker.check(
                                negative_priority, negative_cert,
                                require_complete=False)))

        nonzero_priority = copy.deepcopy(base)
        nonzero_priority["requests"][0]["priority"] = 1
        nonzero_cert = copy.deepcopy(cert)
        rebind(nonzero_priority, nonzero_cert)
        self.assertTrue(any("priority must be zero" in failure
                            for failure in checker.check(
                                nonzero_priority, nonzero_cert,
                                require_complete=False)))

        invalid_profile = copy.deepcopy(base)
        invalid_profile["batch_profiles"]["unused"] = {"x": False}
        invalid_profile_cert = copy.deepcopy(cert)
        rebind(invalid_profile, invalid_profile_cert)
        self.assertTrue(any("batch profile entry" in failure
                            for failure in checker.check(
                                invalid_profile, invalid_profile_cert,
                                require_complete=False)))

        second_server = copy.deepcopy(base)
        second_server["devices"]["SERVER2"] = {"kind": "server", "active_mw": 1}
        second_server_cert = copy.deepcopy(cert)
        rebind(second_server, second_server_cert)
        self.assertTrue(any("only server-kind" in failure
                            for failure in checker.check(
                                second_server, second_server_cert,
                                require_complete=False)))

        bool_peak = copy.deepcopy(cert)
        bool_peak["activation_peak_bytes"] = False
        reseal(bool_peak)
        self.assertTrue(any("activation_peak_bytes" in failure
                            for failure in checker.check(
                                base, bool_peak, require_complete=False)))

        unsupported_route = copy.deepcopy(cert)
        unsupported_route["actions"][0]["device"] = "OP15"
        base_with_phone = copy.deepcopy(base)
        base_with_phone["devices"]["OP15"] = {"kind": "phone", "active_mw": 2000}
        rebind(base_with_phone, unsupported_route)
        route_failures = checker.check(
            base_with_phone, unsupported_route, require_complete=False)
        self.assertTrue(any("uncertified route" in failure
                            for failure in route_failures))

        forged_search = copy.deepcopy(cert)
        forged_search["search"]["states_evaluated"] = 0
        reseal(forged_search)
        failures = checker.check(base, forged_search, require_complete=False)
        self.assertTrue(any("search record fields" in failure for failure in failures))

        incomplete_search = copy.deepcopy(cert)
        incomplete_search["search"]["complete"] = False
        reseal(incomplete_search)
        self.assertTrue(any("search.complete must be true" in failure
                            for failure in checker.check(base, incomplete_search)))

        empty_device = copy.deepcopy(base)
        empty_device["devices"][""] = {"kind": "phone", "active_mw": 1}
        empty_device_cert = copy.deepcopy(cert)
        rebind(empty_device, empty_device_cert)
        self.assertTrue(any("device name" in failure
                            for failure in checker.check(
                                empty_device, empty_device_cert,
                                require_complete=False)))
        with self.assertRaisesRegex(ValueError, "device name"):
            exact.validate_instance(empty_device)

        missing_scope = copy.deepcopy(base)
        missing_scope["evidence"].pop("scope")
        missing_scope_cert = copy.deepcopy(cert)
        rebind(missing_scope, missing_scope_cert)
        self.assertTrue(any("MECHANICS_ONLY" in failure
                            for failure in checker.check(
                                missing_scope, missing_scope_cert,
                                require_complete=False)))
        with self.assertRaisesRegex(ValueError, "MECHANICS_ONLY"):
            exact.validate_instance(missing_scope)

    def test_live_instance_schema_constraints_fail_closed(self):
        base = partial_partition_instance()
        cert = exact.solve(base)
        cases = []

        long_horizon = copy.deepcopy(base)
        long_horizon["horizon_us"] = 1000000000001
        cases.append((long_horizon, "horizon_us"))

        long_id = copy.deepcopy(base)
        long_id["instance_id"] = "i" * 129
        cases.append((long_id, "instance_id"))

        noncanonical_profile = copy.deepcopy(base)
        profile = noncanonical_profile["batch_profiles"]["ffn:w0"]
        profile["016"] = profile["16"]
        cases.append((noncanonical_profile, "016"))

        for invalid, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(ValueError, "instance schema violation"):
                    exact.validate_instance(invalid)
                invalid_cert = copy.deepcopy(cert)
                rebind(invalid, invalid_cert)
                failures = checker.check(
                    invalid, invalid_cert, require_complete=False)
                self.assertTrue(any("instance schema violation" in failure
                                    and diagnostic in failure
                                    for failure in failures), failures)

        boundary = copy.deepcopy(base)
        boundary["horizon_us"] = 1000000000000
        boundary["instance_id"] = "i" * 128
        exact.validate_instance(boundary)

    def test_schema_invalid_instance_skips_independent_optimality_search(self):
        invalid = partial_partition_instance()
        cert = exact.solve(invalid)
        invalid["instance_id"] = "i" * 129
        rebind(invalid, cert)
        with mock.patch.object(
                checker.reference, "optimum",
                side_effect=AssertionError("schema-invalid input reached reference")) \
                as optimum:
            failures = checker.check(invalid, cert, require_complete=True)
        optimum.assert_not_called()
        self.assertTrue(any("instance schema violation" in failure
                            for failure in failures), failures)

    def test_hostile_pythonpath_cannot_shadow_instance_schema_gate(self):
        valid = partial_partition_instance()
        cert = exact.solve(valid)
        invalid = copy.deepcopy(valid)
        invalid["batch_profiles"]["ffn:w0"]["016"] = 545
        rebind(invalid, cert)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            evil = tmp_path / "evil"
            evil.mkdir()
            (evil / "foundation_instance_gate.py").write_text(
                "def instance_schema_errors(_document):\n"
                "    return []\n",
                encoding="ascii",
            )
            instance_path = tmp_path / "invalid.json"
            cert_path = tmp_path / "cert.json"
            instance_path.write_text(json.dumps(invalid), encoding="ascii")
            cert_path.write_text(json.dumps(cert), encoding="ascii")
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{evil}:{ROOT / 'schemas'}"
            oracle_result = subprocess.run(
                [sys.executable, str(ROOT / "oracle" / "exact.py"),
                 "--instance", str(instance_path)],
                capture_output=True, text=True, env=env, check=False,
            )
            checker_result = subprocess.run(
                [sys.executable, str(ROOT / "checker" / "checker.py"),
                 "--instance", str(instance_path),
                 "--certificate", str(cert_path), "--feasibility-only"],
                capture_output=True, text=True, env=env, check=False,
            )
        self.assertNotEqual(oracle_result.returncode, 0, oracle_result.stdout)
        self.assertIn("instance schema violation", oracle_result.stderr)
        self.assertNotEqual(checker_result.returncode, 0, checker_result.stdout)
        self.assertIn("instance schema violation", checker_result.stderr)

    def test_wake_and_idle_entry_boundary_equality(self):
        """Exactly touching P0 windows merge; one microsecond more does not."""
        inst = partial_partition_instance()
        inst["horizon_us"] = 2000
        inst["server_power"] = {"p8_mw": 25000, "p0_mw": 300000, "wake_us": 50,
                                "idle_entry_us": 50, "transition_nj": 1000000}
        # a0 window is [100-50, 200+50] = [50, 250]; a1 window starts at start-50.
        # start == 300 makes the windows touch exactly at 250.
        def pair(second_start):
            return [
                {"id": "a0", "device": "SERVER", "members": ["n0"],
                 "start_us": 100, "finish_us": 200},
                {"id": "a1", "device": "SERVER", "members": ["n1"],
                 "start_us": second_start, "finish_us": second_start + 100},
            ]
        touching = exact._server_energy(inst, pair(300))
        apart = exact._server_energy(inst, pair(301))
        self.assertEqual(touching, checker._server_energy(inst, pair(300)))
        self.assertEqual(apart, checker._server_energy(inst, pair(301)))
        # touching -> one merged window [50,250]+[250,450] and one transition charge
        self.assertEqual(touching[1], [[50, 450]])
        self.assertEqual(len(apart[1]), 2)
        # Identical total active time (400 us) on both sides: the energy gap is
        # purely the extra P8->P0 transition the one-microsecond split introduces.
        self.assertEqual(sum(end - start for start, end in touching[1]), 400)
        self.assertEqual(sum(end - start for start, end in apart[1]), 400)
        self.assertEqual(apart[0] - touching[0], inst["server_power"]["transition_nj"])
        self.assertLess(touching[0], apart[0])

    def test_schedule_mutations_are_rejected(self):
        """horizon, release, deadline, precedence, lane overlap, duration and
        batch-identity mutations must each be caught by the standalone checker."""
        path = ROOT / "fixtures" / "activation_delay_counterexample.json"
        base = json.loads(path.read_text(encoding="ascii"))
        good = exact.solve(base)
        self.assertEqual(checker.check(base, good, require_complete=False), [])

        def mutated_cert(fn):
            cert = copy.deepcopy(good)
            fn(cert)
            reseal(cert)
            return cert

        def by_member(cert, member):
            return next(a for a in cert["actions"] if member in a["members"])

        # precedence: consumer starts before its producer finishes
        precedence = mutated_cert(
            lambda c: by_member(c, "c0").update(
                {"start_us": 0, "finish_us": 6}))
        self.assertTrue(any("before predecessor" in f or "before release" in f
                            for f in checker.check(base, precedence,
                                                   require_complete=False)))

        # duration: finish no longer matches the certified route duration
        duration = mutated_cert(lambda c: by_member(c, "p0").update({"finish_us": 5}))
        self.assertTrue(any("wrong duration" in f
                            for f in checker.check(base, duration,
                                                   require_complete=False)))

        # lane overlap: two SERVER actions overlap on the single server lane
        overlap = mutated_cert(
            lambda c: by_member(c, "p1").update(
                {"start_us": by_member(c, "p0")["start_us"],
                 "finish_us": by_member(c, "p0")["start_us"] + 4}))
        self.assertTrue(any("overlap" in f or "before predecessor" in f
                            for f in checker.check(base, overlap,
                                                   require_complete=False)))

        # release: a node may not start before its own release
        late_release = copy.deepcopy(base)
        for node in late_release["nodes"]:
            if node["id"] == "p0":
                node["release_us"] = 25
        late_cert = copy.deepcopy(good)
        rebind(late_release, late_cert)
        self.assertTrue(any("before release" in f
                            for f in checker.check(late_release, late_cert,
                                                   require_complete=False)))

        # deadline: tightening a deadline must flip the recomputed outcome
        tight = copy.deepcopy(base)
        tight["requests"][0]["deadline_us"] = 5
        tight_cert = copy.deepcopy(good)
        rebind(tight, tight_cert)
        self.assertTrue(any("outcome mismatch" in f
                            for f in checker.check(tight, tight_cert,
                                                   require_complete=False)))

        # horizon: shrinking the horizon must invalidate the finishes
        short = copy.deepcopy(base)
        short["horizon_us"] = 8
        short["requests"][0]["deadline_us"] = 8
        short["requests"][1]["deadline_us"] = 8
        short_cert = copy.deepcopy(good)
        rebind(short, short_cert)
        self.assertTrue(any("invalid finish" in f
                            for f in checker.check(short, short_cert,
                                                   require_complete=False)))

        # batch identity: two different weight sets may not share one server action
        batched = partial_partition_instance()
        batched["nodes"][1]["weight_set_id"] = "w_other"
        cert = exact.solve(partial_partition_instance())
        rebind(batched, cert)
        self.assertTrue(any("illegal batch identity" in f
                            for f in checker.check(batched, cert,
                                                   require_complete=False)))

    def test_unknown_instance_field_fails(self):
        inst = partial_partition_instance()
        cert = exact.solve(inst)
        inst["unexpected"] = 1
        cert["instance_sha256"] = exact.digest(inst)
        reseal(cert)
        self.assertTrue(any(
            "instance fields" in failure
            for failure in checker.check(inst, cert, require_complete=False)
        ))


if __name__ == "__main__":
    unittest.main()
