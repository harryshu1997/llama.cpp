#!/usr/bin/env python3
"""Adversarial tests for the S10-V0-R-E1 typed-evidence gate.

Every test here tries to get an unproven number past the binder. The gate passes
only if all of them fail closed with a stable E_* code.
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "oracle"), str(ROOT / "checker"), str(ROOT / "evidence"),
                str(ROOT / "tests")]
import binder  # noqa: E402
import boundary  # noqa: E402
import canon  # noqa: E402
import exact  # noqa: E402
import checker  # noqa: E402
import make_evidence_fixtures as mk  # noqa: E402
import validator  # noqa: E402

FIX = ROOT / "fixtures" / "evidence"


def load(name):
    return canon.load_strict(str(FIX / name))


def reseal_record(record):
    record["record_sha256"] = canon.record_digest(record)
    return record


def reseal_bundle(bundle):
    bundle["bundle_sha256"] = canon.bundle_digest(bundle)
    return bundle


def find(bundle, record_id):
    for section, kind in validator.RECORD_SECTIONS:
        if kind is None:
            continue
        for record in bundle[section]:
            if record["record_id"] == record_id:
                return record
    raise KeyError(record_id)


def rebind(inst, bundle):
    """Refresh every binding's pinned digest. Used to isolate a mutation.

    Without this, mutating a record would always trip E_HASH first and hide the
    semantic gate under test. Rebinding is exactly what a motivated author would
    do to make a bad bundle look self-consistent, so it is the right adversary.
    """
    records = {}
    for section, kind in validator.RECORD_SECTIONS:
        if kind is None:
            continue
        for record in bundle[section]:
            records[record["record_id"]] = record
    for entry in inst["evidence"]["bindings"]:
        record = records.get(entry["record_id"])
        if record is not None:
            entry["record_sha256"] = record["record_sha256"]
    inst["evidence"]["bundle_sha256"] = bundle["bundle_sha256"]
    return inst


def codes(failures):
    return sorted({failure.split(":")[0] for failure in failures})


class ValidBundle(unittest.TestCase):
    def test_synthetic_mechanics_bundle_is_valid(self):
        self.assertEqual(validator.validate_bundle(load("mechanics_bundle.json")), [])

    def test_both_v3_instances_bind_cleanly(self):
        bundle = load("mechanics_bundle.json")
        for name in ("transition_v3.json", "activation_v3.json"):
            self.assertEqual(validator.validate(load(name), bundle), [], name)

    def test_fixtures_are_deterministic_across_processes(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "make_evidence_fixtures.py"),
             "--check"], capture_output=True, text=True,
            env={"PYTHONHASHSEED": "31337", "PATH": "/usr/bin:/bin"})
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_canonical_bytes_agree_with_frozen_oracle_and_checker(self):
        # The contract requires ONE canonical JSON and SHA-256 implementation.
        # The frozen foundation keeps its own proven code, so prove the bytes are
        # identical rather than asserting it in prose.
        corpus = [load("mechanics_bundle.json"), load("transition_v3.json"),
                  {"a": 1, "b": [1, 2, {"c": "x"}]}, {}, {"z": 0, "a": {"m": -1}}]
        for value in corpus:
            self.assertEqual(canon.canonical(value), exact.canonical(value))
            self.assertEqual(canon.canonical(value), checker.canonical(value))
            self.assertEqual(canon.digest(value), exact.digest(value))
            self.assertEqual(canon.digest(value), checker.digest(value))


class EnumerationIsExhaustive(unittest.TestCase):
    """No integer the solver reads may escape classification.

    This is the gate's single most load-bearing structural property. If some
    evidence-derived field is not enumerated by required_targets(), it is a free
    unbacked number: no binding is demanded for it, so nothing ever checks it.
    Rather than trusting a hand-audit of the solver, every integer leaf of the
    instance is walked and must be either evidence-derived or explicitly declared
    workload/topology. Adding a schema field without classifying it FAILS here.
    """

    # Frozen: policy and topology, chosen by the operator, not measured.
    WORKLOAD_DECLARED = {
        "schema_version", "horizon_us",
        "requests.arrival_us", "requests.deadline_us", "requests.priority",
        "nodes.release_us", "nodes.tokens", "nodes.kv_tokens",
        "evidence.schema_version", "evidence.bindings.value",
    }

    def integer_leaves(self, value, path=""):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            yield path
            return
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{path}.{key}" if path else str(key)
                yield from self.integer_leaves(item, child)
        elif isinstance(value, list):
            for item in value:
                yield from self.integer_leaves(item, path)

    @staticmethod
    def shape_of(path):
        """Collapse instance-specific ids to a structural shape.

        Applied to BOTH the walked instance paths and the enumerated targets, so
        the two are compared on the same footing.
        """
        parts = [p for p in path.split(".") if p]
        if not parts:
            return path
        head = parts[0]
        if head == "nodes":
            return "nodes.routes." + parts[-1] if "routes" in parts \
                else "nodes." + parts[-1]
        if head == "devices":
            return "devices." + parts[-1]
        if head == "requests":
            return "requests." + parts[-1]
        if head == "batch_profiles":
            return "batch_profiles"
        if head == "evidence":
            return "evidence.bindings." + parts[-1] if "bindings" in parts \
                else "evidence." + parts[-1]
        return path

    def test_every_integer_in_the_instance_is_classified(self):
        for name in ("transition_v3.json", "activation_v3.json"):
            inst = load(name)
            derived = {self.shape_of(t) for t in validator.required_targets(inst)}
            self.assertTrue(derived, name)
            for path in self.integer_leaves(inst):
                shape = self.shape_of(path)
                self.assertTrue(
                    shape in derived or shape in self.WORKLOAD_DECLARED,
                    f"{name}: integer at {path} (shape {shape!r}) is neither "
                    f"evidence-derived nor declared workload; an unclassified "
                    f"number is an unbacked number")

    def test_the_guard_itself_catches_an_unclassified_field(self):
        # A guard that cannot fail proves nothing. Inject a new integer field and
        # assert the walk notices it.
        inst = load("transition_v3.json")
        inst["nodes"][0]["smuggled_us"] = 5
        derived = {self.shape_of(t) for t in validator.required_targets(inst)}
        shapes = {self.shape_of(p) for p in self.integer_leaves(inst)}
        self.assertIn("nodes.smuggled_us", shapes)
        self.assertNotIn("nodes.smuggled_us", derived | self.WORKLOAD_DECLARED)

    def test_required_targets_is_frozen_for_the_transition_fixture(self):
        self.assertEqual(sorted(validator.required_targets(load("transition_v3.json"))), [
            "activation_mem_bound_bytes",
            "devices.SERVER.active_mw",
            "nodes.n0.output_bytes",
            "nodes.n0.routes.SERVER.duration_us",
            "nodes.n0.routes.SERVER.extra_energy_nj",
            "nodes.n1.output_bytes",
            "nodes.n1.routes.SERVER.duration_us",
            "nodes.n1.routes.SERVER.extra_energy_nj",
            "server_power.idle_entry_us",
            "server_power.p0_mw",
            "server_power.p8_mw",
            "server_power.transition_nj",
            "server_power.wake_us",
        ])

    def test_every_required_target_has_an_expected_record_kind(self):
        # A target with no expected binding would be reported E_BINDING_EXTRA and
        # could never be satisfied, which would be a silent denial rather than a
        # gate. Every enumerated target must resolve.
        for name in ("transition_v3.json", "activation_v3.json"):
            inst = load(name)
            for target in validator.required_targets(inst):
                kind, field, _device = validator._expected_binding(target, inst)
                self.assertIsNotNone(kind, f"{name}: {target} has no expected kind")
                self.assertIsNotNone(field, f"{name}: {target} has no expected field")


class ProjectionCarriesNoUnboundNumber(unittest.TestCase):
    def test_projection_targets_equal_instance_targets(self):
        for name in ("transition_v3.json", "activation_v3.json"):
            inst = load(name)
            core = binder.project_v2(inst)
            self.assertEqual(validator.required_targets(core),
                             validator.required_targets(inst), name)

    def test_solver_reads_no_field_the_projection_did_not_carry(self):
        # exact.validate_instance enforces the exact v2 field set, so a projection
        # that grew or lost a field cannot even be solved.
        for name in ("transition_v3.json", "activation_v3.json"):
            core = binder.project_v2(load(name))
            exact.validate_instance(core)


class FrozenOptimaThroughEvidence(unittest.TestCase):
    """The evidence path must reproduce the frozen temporal optima exactly."""

    def test_transition_v3_selects_147250000(self):
        inst, bundle = load("transition_v3.json"), load("mechanics_bundle.json")
        cert = binder.solve_bound(inst, bundle)
        self.assertEqual(cert["objective"], [0, 0, -2, 147250000])
        self.assertEqual(cert["energy"]["server_p0_intervals"], [[800, 1150]])
        self.assertEqual(cert["evidence"]["energy_claim"], "NONE_MECHANICS_ONLY")
        self.assertEqual(binder.check_bound(inst, bundle, cert), [])

    def test_activation_v3_selects_2974000(self):
        inst, bundle = load("activation_v3.json"), load("mechanics_bundle.json")
        cert = binder.solve_bound(inst, bundle)
        self.assertEqual(cert["objective"], [0, 0, -2, 2974000])
        self.assertEqual(cert["activation_peak_bytes"], 100)
        self.assertEqual(binder.check_bound(inst, bundle, cert), [])

    def test_projection_matches_the_frozen_v2_mechanics(self):
        # The v3 core must be the frozen v2 fixture apart from the evidence label,
        # so "the evidence path reproduces the frozen optimum" is not a coincidence.
        core = binder.project_v2(load("transition_v3.json"))
        frozen = canon.load_strict(
            str(ROOT / "fixtures" / "transition_delay_counterexample.json"))
        for field in ("horizon_us", "activation_mem_bound_bytes", "server_power",
                      "devices", "batch_profiles"):
            self.assertEqual(core[field], frozen[field], field)
        self.assertEqual([{k: v for k, v in n.items()} for n in core["nodes"]],
                         [{k: v for k, v in n.items()} for n in frozen["nodes"]])


class DigestAndSwapMutations(unittest.TestCase):
    def setUp(self):
        self.bundle = load("mechanics_bundle.json")
        self.inst = load("transition_v3.json")

    def assert_code(self, code, inst=None, bundle=None):
        failures = validator.validate(inst or self.inst, bundle or self.bundle)
        self.assertIn(code, codes(failures), failures)
        return failures

    def test_record_edited_without_resealing_is_caught(self):
        find(self.bundle, "route.n0.server")["latency"]["p95_us"] = 1
        self.assert_code("E_HASH")

    def test_record_edited_and_resealed_without_bundle_reseal_is_caught(self):
        record = find(self.bundle, "route.n0.server")
        record["latency"]["p95_us"] = 1
        reseal_record(record)
        self.assert_code("E_HASH")

    def test_bundle_mutation_without_digest_update_is_caught(self):
        self.bundle["bundle_id"] = "swapped"
        self.assert_code("E_HASH")

    def test_instance_bound_to_a_different_bundle_is_caught(self):
        self.inst["evidence"]["bundle_sha256"] = "f" * 64
        self.assert_code("E_HASH")

    def test_binding_pinning_a_stale_record_digest_is_caught(self):
        self.inst["evidence"]["bindings"][1]["record_sha256"] = "a" * 64
        self.assert_code("E_HASH")

    def test_artifact_digest_swap_is_caught(self):
        # Swapping the artifact bytes changes the artifact record, hence the
        # bundle digest.
        self.bundle["artifacts"][0]["sha256"] = "b" * 64
        self.assert_code("E_HASH")

    def test_resealed_artifact_digest_is_checked_against_file_bytes(self):
        self.bundle["artifacts"][0]["sha256"] = "b" * 64
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assert_code("E_ARTIFACT_HASH")

    def test_source_revision_disagreement_within_one_record_is_stale(self):
        # A record whose own artifacts disagree about the source revision cannot
        # name a single thing that was measured.
        extra = copy.deepcopy(self.bundle["artifacts"][0])
        extra["artifact_id"] = "art.server.trace.other"
        extra["source_revision"] = "deadbeef"
        self.bundle["artifacts"].append(extra)
        record = find(self.bundle, "route.n0.server")
        record["artifacts"] = ["art.server.trace", "art.server.trace.other"]
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assert_code("E_STALE")

    def test_build_revision_swap_is_stale(self):
        self.bundle["artifacts"][0]["backend_build"] = "other-build"
        reseal_bundle(self.bundle)
        self.assert_code("E_STALE")

    def test_device_swap_between_record_and_artifact_is_stale(self):
        self.bundle["artifacts"][0]["device_id"] = "OP12"
        reseal_bundle(self.bundle)
        self.assert_code("E_STALE")

    def test_artifact_expired_at_evaluation_epoch_is_stale(self):
        for artifact in self.bundle["artifacts"]:
            artifact["timestamp_utc"] = "20000101T000000Z"
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assert_code("E_STALE")

    def test_artifact_from_after_evaluation_epoch_is_stale(self):
        self.bundle["artifacts"][0]["timestamp_utc"] = "20260715T000001Z"
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = self.assert_code("E_STALE")
        self.assertTrue(any("after bundle evaluation" in failure
                            for failure in failures), failures)

    def test_artifact_requires_positive_validity_in_schema_and_runtime(self):
        artifact = self.bundle["artifacts"][0]
        artifact["validity_us"] = 0
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assert_code("E_SCHEMA")
        failures = []
        validator._check_artifact_validity(
            self.bundle, {artifact["artifact_id"]: artifact}, failures)
        self.assertIn("E_STALE", codes(failures))

    def test_evaluation_epoch_must_be_a_real_utc_timestamp(self):
        self.bundle["evaluation_timestamp_utc"] = "20260230T000000Z"
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assert_code("E_STALE")

    def test_duplicate_record_id_is_caught(self):
        self.bundle["routes"].append(copy.deepcopy(self.bundle["routes"][0]))
        reseal_bundle(self.bundle)
        self.assert_code("E_DUPLICATE_ID")

    def test_duplicate_artifact_id_is_caught(self):
        self.bundle["artifacts"].append(copy.deepcopy(self.bundle["artifacts"][0]))
        reseal_bundle(self.bundle)
        self.assert_code("E_DUPLICATE_ID")

    def test_missing_record_is_caught(self):
        self.bundle["routes"] = [r for r in self.bundle["routes"]
                                 if r["record_id"] != "route.n0.server"]
        reseal_bundle(self.bundle)
        self.inst["evidence"]["bundle_sha256"] = self.bundle["bundle_sha256"]
        self.assert_code("E_MISSING_RECORD")

    def test_missing_artifact_is_caught(self):
        self.bundle["artifacts"] = [a for a in self.bundle["artifacts"]
                                    if a["artifact_id"] != "art.server.trace"]
        reseal_bundle(self.bundle)
        self.assert_code("E_MISSING_ARTIFACT")

    def test_empty_artifact_list_is_caught(self):
        record = find(self.bundle, "route.n0.server")
        record["artifacts"] = []
        reseal_record(record)
        reseal_bundle(self.bundle)
        self.assert_code("E_EMPTY_ARTIFACTS")


class IdentityRebindMutations(unittest.TestCase):
    """A record must not be reusable for a different configuration."""

    def setUp(self):
        self.bundle = load("mechanics_bundle.json")
        self.inst = load("transition_v3.json")

    def mutate_route(self, field, value):
        record = find(self.bundle, "route.n0.server")
        record[field] = value
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        return validator.validate(self.inst, self.bundle)

    def test_model_swap_is_caught(self):
        self.assertIn("E_IDENTITY", codes(self.mutate_route("model_id", "m_other")))

    def test_weight_set_swap_is_caught(self):
        self.assertIn("E_IDENTITY",
                      codes(self.mutate_route("weight_set_id", "w_other")))

    def test_graph_swap_is_caught(self):
        self.assertIn("E_IDENTITY", codes(self.mutate_route("graph_id", "graph.other")))

    def test_island_swap_is_caught(self):
        self.assertIn("E_IDENTITY",
                      codes(self.mutate_route("island_id", "island.other")))

    def test_kernel_swap_against_correctness_is_caught(self):
        self.assertIn("E_IDENTITY",
                      codes(self.mutate_route("kernel_id", "other_kernel")))

    def test_device_swap_is_caught(self):
        failures = self.mutate_route("device_id", "OP15")
        self.assertTrue({"E_IDENTITY", "E_STALE"} & set(codes(failures)), failures)

    def test_shape_envelope_swap_is_caught(self):
        self.assertIn("E_ENVELOPE",
                      codes(self.mutate_route("shape_envelope",
                                              {"tokens_min": 4, "tokens_max": 8,
                                               "kv_min": 0, "kv_max": 10})))

    def test_batch_size_swap_is_caught(self):
        self.assertIn("E_ENVELOPE", codes(self.mutate_route("batch_size", 4)))

    def test_thermal_record_from_another_device_is_caught(self):
        self.assertIn("E_THERMAL", codes(self.mutate_route("thermal_id", "thr.op15")))

    def test_binding_moved_to_a_record_of_the_wrong_kind_is_caught(self):
        entry = next(b for b in self.inst["evidence"]["bindings"]
                     if b["target"] == "nodes.n0.routes.SERVER.duration_us")
        entry["record_id"] = "pwr.server"
        entry["record_sha256"] = find(self.bundle, "pwr.server")["record_sha256"]
        self.assertIn("E_IDENTITY", codes(validator.validate(self.inst, self.bundle)))

    def test_power_record_of_another_device_cannot_serve_the_server(self):
        entry = next(b for b in self.inst["evidence"]["bindings"]
                     if b["target"] == "devices.SERVER.active_mw")
        record = find(self.bundle, "pwr.op15")
        record["active_mw"] = 300000
        reseal_record(record)
        reseal_bundle(self.bundle)
        entry["record_id"] = "pwr.op15"
        entry["record_sha256"] = record["record_sha256"]
        rebind(self.inst, self.bundle)
        self.assertIn("E_IDENTITY", codes(validator.validate(self.inst, self.bundle)))


class EligibilityMutations(unittest.TestCase):
    def setUp(self):
        self.bundle = load("mechanics_bundle.json")
        self.inst = load("transition_v3.json")

    def mutate(self, record_id, field, value):
        record = find(self.bundle, record_id)
        record[field] = value
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        return validator.validate(self.inst, self.bundle)

    def test_pass_route_with_failed_correctness_is_caught(self):
        self.assertIn("E_CORRECTNESS",
                      codes(self.mutate("corr.m0.server", "status", "FAIL")))

    def test_pass_route_with_unknown_correctness_is_caught(self):
        self.assertIn("E_CORRECTNESS",
                      codes(self.mutate("corr.m0.server", "status", "UNKNOWN")))

    def test_correctness_pass_with_failing_verdict_is_caught(self):
        self.assertIn("E_CORRECTNESS",
                      codes(self.mutate("corr.m0.server", "verdict", "FAIL")))

    def test_correctness_above_its_own_threshold_is_caught(self):
        self.assertIn("E_CORRECTNESS",
                      codes(self.mutate("corr.m0.server", "observed_ppm", 900000)))

    def test_exact_correctness_requires_zero_error_and_threshold(self):
        record = find(self.bundle, "corr.m0.server")
        record["metric"] = "EXACT"
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_SCHEMA", codes(validator.validate(self.inst, self.bundle)))
        semantic_failures = []
        validator._check_correctness(record, semantic_failures)
        self.assertIn("E_CORRECTNESS", codes(semantic_failures))

    def test_cpu_fallback_observed_is_caught(self):
        failures = self.mutate("corr.m0.server", "no_fallback_proof",
                               {"method": "OP_TRACE", "artifact_id": "art.server.trace",
                                "fallback_ops_observed": 1})
        self.assertIn("E_FALLBACK", codes(failures))

    def test_missing_no_fallback_proof_is_caught(self):
        failures = self.mutate("corr.m0.server", "no_fallback_proof",
                               {"method": "NONE", "artifact_id": "art.server.trace",
                                "fallback_ops_observed": 0})
        self.assertIn("E_FALLBACK", codes(failures))

    def test_too_few_processes_is_caught(self):
        self.assertIn("E_SAMPLES",
                      codes(self.mutate("route.n0.server", "process_count", 1)))

    def test_too_few_samples_is_caught(self):
        self.assertIn("E_SAMPLES",
                      codes(self.mutate("route.n0.server", "sample_count", 2)))

    def test_too_few_power_samples_is_caught(self):
        self.assertIn("E_SAMPLES",
                      codes(self.mutate("pwr.server", "sample_count", 57)))

    def test_expired_validity_is_caught(self):
        self.assertIn("E_STALE", codes(self.mutate("thr.server", "validity_us", 0)))

    def test_unknown_thermal_state_is_caught(self):
        self.assertIn("E_THERMAL",
                      codes(self.mutate("thr.server", "thermal_state", "UNKNOWN")))

    def test_ineligible_record_cannot_be_bound(self):
        self.assertIn("E_STATUS",
                      codes(self.mutate("route.n0.server", "status", "INELIGIBLE")))

    def test_non_monotone_latency_is_caught(self):
        self.assertIn("E_SCHEMA",
                      codes(self.mutate("route.n0.server", "latency",
                                        {"p50_us": 900, "p95_us": 100, "max_us": 50})))

    def test_active_below_idle_power_is_caught(self):
        self.assertIn("E_SCHEMA", codes(self.mutate("pwr.server", "idle_mw", 400000)))

    def test_correctness_compared_against_itself_is_caught(self):
        failures = self.mutate("corr.m0.server", "reference_route",
                               {"reference_kind": "CPU_FP32",
                                "device_id": "SERVER",
                                "backend_build": mk.BUILD,
                                "kernel_id": "synthetic_kernel_server",
                                "artifact_id": "art.server.trace"})
        self.assertIn("E_SCHEMA", codes(failures))


class DerivedValueMutations(unittest.TestCase):
    """Off-by-one in any derived value must fail, in both directions."""

    def setUp(self):
        self.bundle = load("mechanics_bundle.json")
        self.inst = load("transition_v3.json")

    def test_instance_duration_one_unit_off_is_caught(self):
        self.inst["nodes"][0]["routes"]["SERVER"]["duration_us"] = 101
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_BINDING_VALUE", codes(failures))

    def test_instance_and_binding_agree_but_record_differs_is_caught(self):
        # The strongest form: the instance and its binding are self-consistent and
        # only the hashed record disagrees.
        self.inst["nodes"][0]["routes"]["SERVER"]["duration_us"] = 101
        entry = next(b for b in self.inst["evidence"]["bindings"]
                     if b["target"] == "nodes.n0.routes.SERVER.duration_us")
        entry["value"] = 101
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_BINDING_VALUE", codes(failures))

    def test_power_one_unit_off_is_caught(self):
        self.inst["server_power"]["p0_mw"] = 300001
        self.inst["devices"]["SERVER"]["active_mw"] = 300001
        self.assertIn("E_BINDING_VALUE",
                      codes(validator.validate(self.inst, self.bundle)))

    def test_transition_energy_one_unit_off_is_caught(self):
        self.inst["server_power"]["transition_nj"] = 999999
        self.assertIn("E_BINDING_VALUE",
                      codes(validator.validate(self.inst, self.bundle)))

    def test_output_bytes_one_unit_off_is_caught(self):
        inst = load("activation_v3.json")
        inst["nodes"][0]["output_bytes"] = 101
        self.assertIn("E_BINDING_VALUE", codes(validator.validate(inst, self.bundle)))

    def test_activation_bound_one_unit_off_is_caught(self):
        inst = load("activation_v3.json")
        inst["activation_mem_bound_bytes"] = 151
        self.assertIn("E_BINDING_VALUE", codes(validator.validate(inst, self.bundle)))

    def test_batch_duration_one_unit_off_is_caught(self):
        inst = load("transition_v3.json")
        inst["batch_profiles"] = {"bk": {"1": 100}}
        inst["nodes"][0]["batch_key"] = "bk"
        rebound = next(b for b in inst["evidence"]["bindings"]
                       if b["target"] == "nodes.n0.routes.SERVER.duration_us")
        inst["evidence"]["bindings"].append(
            {"target": "batch_profiles.bk.1", "record_id": rebound["record_id"],
             "record_sha256": rebound["record_sha256"], "field": "latency.p95_us",
             "value": 100})
        self.assertEqual(validator.validate(inst, self.bundle), [])
        inst["batch_profiles"]["bk"]["1"] = 101
        self.assertIn("E_BINDING_VALUE", codes(validator.validate(inst, self.bundle)))

    def test_missing_binding_fails_closed(self):
        self.inst["evidence"]["bindings"] = [
            b for b in self.inst["evidence"]["bindings"]
            if b["target"] != "server_power.transition_nj"]
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_BINDING_MISSING", codes(failures))

    def test_dropping_every_binding_fails_closed(self):
        # Required targets come from the instance, so an empty binding list is the
        # loudest possible failure rather than a silently unchecked instance.
        self.inst["evidence"]["bindings"] = []
        failures = validator.validate(self.inst, self.bundle)
        self.assertEqual(codes(failures), ["E_BINDING_MISSING"])
        self.assertEqual(len(failures), len(validator.required_targets(self.inst)))

    def test_extra_binding_for_a_non_evidence_field_is_caught(self):
        self.inst["evidence"]["bindings"].append(
            {"target": "horizon_us", "record_id": "pwr.server",
             "record_sha256": find(self.bundle, "pwr.server")["record_sha256"],
             "field": "wake_us", "value": 2000})
        self.assertIn("E_BINDING_EXTRA",
                      codes(validator.validate(self.inst, self.bundle)))

    def test_binding_to_the_wrong_field_of_the_right_record_is_caught(self):
        entry = next(b for b in self.inst["evidence"]["bindings"]
                     if b["target"] == "server_power.p8_mw")
        entry["field"] = "active_mw"
        self.assertIn("E_BINDING_VALUE",
                      codes(validator.validate(self.inst, self.bundle)))

    def test_target_bound_twice_is_caught(self):
        entry = copy.deepcopy(self.inst["evidence"]["bindings"][0])
        self.inst["evidence"]["bindings"].append(entry)
        self.assertIn("E_SCHEMA", codes(validator.validate(self.inst, self.bundle)))


class EvidenceRepairRegressions(unittest.TestCase):
    """Fail-open paths found after the worker's original E1 green report."""

    def setUp(self):
        self.bundle = load("mechanics_bundle.json")
        self.inst = load("transition_v3.json")

    def make_two_member_batch(self):
        nodes = self.inst["nodes"]
        for node in nodes:
            node["batch_key"] = "mix"
            node["output_bytes"] = 100
        nodes[1]["model_id"] = nodes[0]["model_id"]
        nodes[1]["weight_set_id"] = nodes[0]["weight_set_id"]
        nodes[1]["graph_id"] = nodes[0]["graph_id"]

        boundary_n0 = find(self.bundle, "bnd.n0")
        boundary_n1 = find(self.bundle, "bnd.n1")
        boundary_n0["output_bytes"] = 100
        boundary_n1["output_bytes"] = 100
        for field in ("model_id", "weight_set_id", "graph_id", "island_id"):
            boundary_n1[field] = nodes[1][field]
        reseal_record(boundary_n0)
        reseal_record(boundary_n1)

        route_n1 = find(self.bundle, "route.n1.server")
        for field in ("model_id", "weight_set_id", "graph_id", "island_id"):
            route_n1[field] = nodes[1][field]
        route_n1["correctness_id"] = "corr.m0.server"
        reseal_record(route_n1)

        batch_boundary = copy.deepcopy(boundary_n0)
        batch_boundary["record_id"] = "bnd.batch.mix.2"
        batch_boundary["output_bytes"] = 0
        reseal_record(batch_boundary)
        self.bundle["boundaries"].append(batch_boundary)

        batch_route = copy.deepcopy(find(self.bundle, "route.n0.server"))
        batch_route["record_id"] = "route.batch.mix.2"
        batch_route["boundary_id"] = batch_boundary["record_id"]
        batch_route["batch_size"] = 2
        batch_route["latency"] = {"p50_us": 1, "p95_us": 1, "max_us": 2}
        reseal_record(batch_route)
        self.bundle["routes"].append(batch_route)

        self.inst["batch_profiles"] = {"mix": {"2": 1}}
        for binding in self.inst["evidence"]["bindings"]:
            if binding["target"] in ("nodes.n0.output_bytes", "nodes.n1.output_bytes"):
                binding["value"] = 100
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.inst["evidence"]["bindings"].append(mk.make_binding(
            self.bundle, "batch_profiles.mix.2", batch_route["record_id"],
            "latency.p95_us", 1))
        return batch_boundary

    def test_live_schema_rejects_resealed_unknown_record_field(self):
        record = find(self.bundle, "route.n0.server")
        record["forged"] = 1
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_SCHEMA", codes(validator.validate(self.inst, self.bundle)))

    def test_live_schema_rejects_unknown_instance_field(self):
        self.inst["forged"] = 1
        self.assertIn("E_SCHEMA", codes(validator.validate(self.inst, self.bundle)))

    def test_live_schema_rejects_missing_artifact_descriptor_field(self):
        del self.bundle["artifacts"][0]["path"]
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_SCHEMA", codes(validator.validate(self.inst, self.bundle)))

    def test_schema_engine_failure_is_a_refusal(self):
        original = validator.SYSTEM_PYTHON
        validator.SYSTEM_PYTHON = pathlib.Path("/definitely/missing/python")
        try:
            failures = validator.validate(self.inst, self.bundle)
        finally:
            validator.SYSTEM_PYTHON = original
        self.assertIn("E_SCHEMA_ENGINE", codes(failures))

    def test_schema_gate_ignores_hostile_pythonpath(self):
        record = find(self.bundle, "route.n0.server")
        record["forged"] = 1
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "jsonschema.py").write_text(
                "class Draft202012Validator:\n"
                "    check_schema = staticmethod(lambda schema: None)\n"
                "    def __init__(self, schema): pass\n"
                "    def iter_errors(self, document): return []\n",
                encoding="ascii",
            )
            old = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = tmp
            try:
                failures = validator.validate(self.inst, self.bundle)
            finally:
                if old is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = old
        self.assertIn("E_SCHEMA", codes(failures))

    def test_artifact_path_traversal_is_rejected(self):
        self.bundle["artifacts"][0]["path"] = "../outside.json"
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_ARTIFACT_PATH",
                      codes(validator.validate(self.inst, self.bundle)))

    def test_missing_artifact_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shutil.copytree(FIX / "synthetic", root / "synthetic")
            (root / self.bundle["artifacts"][0]["path"]).unlink()
            failures = validator.validate(self.inst, self.bundle, root)
        self.assertIn("E_ARTIFACT_MISSING", codes(failures))

    def test_no_fallback_proof_must_name_a_listed_artifact(self):
        record = find(self.bundle, "corr.m0.server")
        record["no_fallback_proof"]["artifact_id"] = "artifact.phantom"
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_MISSING_ARTIFACT", codes(failures))
        self.assertIn("E_FALLBACK", codes(failures))

    def test_no_fallback_proof_and_cpu_reference_cannot_alias_one_payload(self):
        proof = next(artifact for artifact in self.bundle["artifacts"]
                     if artifact["artifact_id"] == "art.server.trace")
        reference = next(artifact for artifact in self.bundle["artifacts"]
                         if artifact["artifact_id"] == "art.reference.cpu")
        reference["path"] = proof["path"]
        reference["sha256"] = proof["sha256"]
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_FALLBACK", codes(failures))
        self.assertTrue(any("same path or payload" in failure
                            for failure in failures), failures)

    def test_route_and_correctness_artifact_revision_sets_must_match(self):
        target = copy.deepcopy(next(
            artifact for artifact in self.bundle["artifacts"]
            if artifact["artifact_id"] == "art.server.trace"))
        reference = copy.deepcopy(next(
            artifact for artifact in self.bundle["artifacts"]
            if artifact["artifact_id"] == "art.reference.cpu"))
        target["artifact_id"] = "art.server.trace.stale.correctness"
        reference["artifact_id"] = "art.reference.cpu.stale.correctness"
        target["source_revision"] = "stale-correctness-source"
        reference["source_revision"] = "stale-correctness-source"
        self.bundle["artifacts"].extend((target, reference))
        record = find(self.bundle, "corr.m0.server")
        record["artifacts"] = [target["artifact_id"], reference["artifact_id"]]
        record["no_fallback_proof"]["artifact_id"] = target["artifact_id"]
        record["reference_route"]["artifact_id"] = reference["artifact_id"]
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_STALE", codes(failures))
        self.assertTrue(any("and correctness" in failure and
                            "source_revision" in failure for failure in failures),
                        failures)

    def test_route_and_thermal_artifact_revision_sets_must_match(self):
        artifact = copy.deepcopy(next(
            item for item in self.bundle["artifacts"]
            if item["artifact_id"] == "art.server.trace"))
        artifact["artifact_id"] = "art.server.trace.stale.thermal"
        artifact["build_revision"] = "stale-thermal-build"
        self.bundle["artifacts"].append(artifact)
        record = find(self.bundle, "thr.server")
        record["artifacts"] = [artifact["artifact_id"]]
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_STALE", codes(failures))
        self.assertTrue(any("and thermal profile" in failure and
                            "build_revision" in failure for failure in failures),
                        failures)

    def test_thermal_artifact_backend_build_must_match_route(self):
        artifact = copy.deepcopy(next(
            item for item in self.bundle["artifacts"]
            if item["artifact_id"] == "art.server.trace"))
        artifact["artifact_id"] = "art.server.trace.other.thermal.backend"
        artifact["backend_build"] = "different-backend-binary"
        self.bundle["artifacts"].append(artifact)
        record = find(self.bundle, "thr.server")
        record["artifacts"] = [artifact["artifact_id"]]
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_STALE", codes(failures))
        self.assertTrue(any("thermal" in failure and "backend build" in failure
                            for failure in failures), failures)

    def test_thermal_observation_duration_must_be_positive(self):
        record = find(self.bundle, "thr.server")
        record["duration_us"] = 0
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_SCHEMA", codes(validator.validate(self.inst, self.bundle)))
        failures = []
        validator._check_thermal(record, failures)
        self.assertIn("E_THERMAL", codes(failures))

    def test_activation_capacity_probe_must_be_server_owned(self):
        record = find(self.bundle, "route.a.c0.op15")
        record["layer_class"] = validator.CAPACITY_LAYER_CLASS
        reseal_record(record)
        reseal_bundle(self.bundle)
        binding = next(entry for entry in self.inst["evidence"]["bindings"]
                       if entry["target"] == "activation_mem_bound_bytes")
        binding["record_id"] = record["record_id"]
        binding["record_sha256"] = record["record_sha256"]
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_IDENTITY", codes(failures))
        self.assertTrue(any("concerns device SERVER" in failure or
                            "SERVER-owned capacity" in failure
                            for failure in failures), failures)

    def test_transport_domain_none_requires_zero_bytes_and_time(self):
        boundary_record = find(self.bundle, "bnd.n0")
        route_record = find(self.bundle, "route.n0.server")
        boundary_record["h2d_bytes"] = 64
        route_record["h2d_bytes"] = 64
        reseal_record(boundary_record)
        reseal_record(route_record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_INCOHERENT", codes(failures))
        self.assertTrue(any("transport_domain NONE" in failure and
                            "not all zero" in failure for failure in failures),
                        failures)

    def test_duplicate_node_ids_are_rejected_before_binding_projection(self):
        self.inst["nodes"].append(copy.deepcopy(self.inst["nodes"][0]))
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_DUPLICATE_ID", codes(failures))
        self.assertTrue(any("node ids are repeated" in failure
                            for failure in failures), failures)

    def test_node_kv_must_be_inside_route_envelope(self):
        self.inst["nodes"][0]["kv_tokens"] = 4097
        self.assertIn("E_ENVELOPE", codes(validator.validate(self.inst, self.bundle)))

    def test_correctness_envelope_must_cover_route_kv(self):
        route = find(self.bundle, "route.n0.server")
        route["shape_envelope"]["kv_min"] = 1000
        route["shape_envelope"]["kv_max"] = 1000
        cert = find(self.bundle, "corr.m0.server")
        cert["shape_envelope"]["kv_min"] = 0
        cert["shape_envelope"]["kv_max"] = 0
        self.inst["nodes"][0]["kv_tokens"] = 1000
        reseal_record(route)
        reseal_record(cert)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_ENVELOPE", codes(validator.validate(self.inst, self.bundle)))

    def test_route_pins_the_boundary_used_for_energy(self):
        cheap = copy.deepcopy(find(self.bundle, "bnd.n0"))
        cheap["record_id"] = "bnd.n0.cheap"
        cheap["energy_nj"] = 1
        reseal_record(cheap)
        self.bundle["boundaries"].append(cheap)
        reseal_bundle(self.bundle)
        self.inst["nodes"][0]["routes"]["SERVER"]["extra_energy_nj"] = 1
        binding = next(entry for entry in self.inst["evidence"]["bindings"]
                       if entry["target"] ==
                       "nodes.n0.routes.SERVER.extra_energy_nj")
        binding["record_id"] = cheap["record_id"]
        binding["record_sha256"] = cheap["record_sha256"]
        binding["value"] = 1
        rebind(self.inst, self.bundle)
        self.assertIn("E_INCOHERENT",
                      codes(validator.validate(self.inst, self.bundle)))

    def test_overlapping_additive_rails_are_rejected(self):
        inst = load("activation_v3.json")
        phone = find(self.bundle, "pwr.op15")
        phone["included_rails"].append("SERVER/CPU")
        reseal_record(phone)
        reseal_bundle(self.bundle)
        rebind(inst, self.bundle)
        self.assertIn("E_DOUBLE_COUNT", codes(validator.validate(inst, self.bundle)))

    def test_resealed_certificate_with_unknown_field_is_rejected(self):
        cert = binder.solve_bound(self.inst, self.bundle)
        cert["forged"] = 1
        cert["certificate_sha256"] = canon.digest(
            {key: value for key, value in cert.items()
             if key != "certificate_sha256"})
        self.assertIn("E_SCHEMA", codes(binder.check_bound(
            self.inst, self.bundle, cert)))

    def test_resealed_incomplete_search_certificate_is_rejected(self):
        cert = binder.solve_bound(self.inst, self.bundle)
        cert["search"]["complete"] = False
        cert["certificate_sha256"] = canon.digest(
            {key: value for key, value in cert.items()
             if key != "certificate_sha256"})
        self.assertIn("E_SCHEMA", codes(binder.check_bound(
            self.inst, self.bundle, cert)))

    def test_boundary_and_route_source_revisions_must_match(self):
        artifact = next(item for item in self.bundle["artifacts"]
                        if item["artifact_id"] == "art.boundary")
        artifact["source_revision"] = "old-source"
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_STALE", codes(validator.validate(self.inst, self.bundle)))

    def test_route_end_to_end_latency_cannot_omit_boundary_wall_time(self):
        record = find(self.bundle, "bnd.n0")
        record["transfer_us"] = 1000
        record["boundary_wall_us"] = 1000
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_INCOHERENT",
                      codes(validator.validate(self.inst, self.bundle)))

    def test_positive_boundary_energy_needs_a_nonzero_window(self):
        record = find(self.bundle, "bnd.n0")
        record["energy_nj"] = 1
        reseal_record(record)
        reseal_bundle(self.bundle)
        self.inst["nodes"][0]["routes"]["SERVER"]["extra_energy_nj"] = 1
        binding = next(entry for entry in self.inst["evidence"]["bindings"]
                       if entry["target"] ==
                       "nodes.n0.routes.SERVER.extra_energy_nj")
        binding["value"] = 1
        rebind(self.inst, self.bundle)
        self.assertIn("E_INCOHERENT",
                      codes(validator.validate(self.inst, self.bundle)))

    def test_nonzero_usb_bytes_need_nonzero_transfer_time(self):
        inst = load("activation_v3.json")
        boundary_record = find(self.bundle, "bnd.a.c0")
        route_record = find(self.bundle, "route.a.c0.op15")
        boundary_record["h2d_bytes"] = 64
        boundary_record["d2h_bytes"] = 32
        boundary_record["output_bytes"] = 64
        route_record["h2d_bytes"] = 64
        route_record["d2h_bytes"] = 32
        inst["nodes"][1]["output_bytes"] = 64
        binding = next(entry for entry in inst["evidence"]["bindings"]
                       if entry["target"] == "nodes.c0.output_bytes")
        binding["value"] = 64
        reseal_record(boundary_record)
        reseal_record(route_record)
        reseal_bundle(self.bundle)
        rebind(inst, self.bundle)
        failures = validator.validate(inst, self.bundle)
        self.assertTrue(any("zero transfer/wall time" in failure
                            for failure in failures), failures)

    def test_phone_route_bytes_must_match_its_boundary(self):
        inst = load("activation_v3.json")
        boundary_record = find(self.bundle, "bnd.a.c0")
        boundary_record["h2d_bytes"] = 64
        boundary_record["d2h_bytes"] = 32
        boundary_record["output_bytes"] = 64
        boundary_record["transfer_us"] = 1
        boundary_record["boundary_wall_us"] = 1
        inst["nodes"][1]["output_bytes"] = 64
        binding = next(entry for entry in inst["evidence"]["bindings"]
                       if entry["target"] == "nodes.c0.output_bytes")
        binding["value"] = 64
        reseal_record(boundary_record)
        reseal_bundle(self.bundle)
        rebind(inst, self.bundle)
        failures = validator.validate(inst, self.bundle)
        self.assertTrue(any("declares h2d/d2h bytes" in failure
                            for failure in failures), failures)

    def test_intra_host_route_bytes_must_match_its_boundary(self):
        route_record = find(self.bundle, "route.n0.server")
        route_record["h2d_bytes"] = 64
        reseal_record(route_record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertTrue(any("declares h2d/d2h bytes" in failure
                            for failure in failures), failures)

    def test_batch_boundary_must_match_combined_member_output(self):
        self.make_two_member_batch()
        failures = validator.validate(self.inst, self.bundle)
        self.assertTrue(any("legal batch geometry produces 200 bytes" in failure
                            for failure in failures), failures)

    def test_batch_boundary_direction_must_be_server_compatible(self):
        boundary_record = self.make_two_member_batch()
        boundary_record["output_bytes"] = 200
        boundary_record["direction"] = "HOST_TO_PHONE"
        reseal_record(boundary_record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertTrue(any("is a SERVER batch" in failure
                            for failure in failures), failures)

    def test_batch_boundary_energy_cannot_be_silently_dropped(self):
        boundary_record = self.make_two_member_batch()
        boundary_record["output_bytes"] = 200
        boundary_record["energy_nj"] = 1
        boundary_record["boundary_wall_us"] = 1
        reseal_record(boundary_record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertTrue(any("no batch-boundary energy term" in failure
                            for failure in failures), failures)

    def test_reference_artifact_must_exist_and_be_listed(self):
        record = find(self.bundle, "corr.m0.server")
        record["reference_route"]["artifact_id"] = "artifact.phantom.reference"
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_MISSING_ARTIFACT", codes(failures))
        self.assertIn("E_CORRECTNESS", codes(failures))

    def test_cpu_reference_artifact_cannot_prove_target_no_fallback(self):
        record = find(self.bundle, "corr.m0.server")
        record["no_fallback_proof"]["artifact_id"] = "art.reference.cpu"
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_FALLBACK",
                      codes(validator.validate(self.inst, self.bundle)))

    def test_schedulable_route_requires_accelerated_correctness(self):
        record = find(self.bundle, "corr.m0.server")
        record["route_kind"] = "REFERENCE"
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_CORRECTNESS",
                      codes(validator.validate(self.inst, self.bundle)))

    def test_active_only_phone_profile_cannot_hide_transition_cost(self):
        inst = load("activation_v3.json")
        record = find(self.bundle, "pwr.op15")
        record["wake_us"] = 1
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(inst, self.bundle)
        self.assertIn("E_INCOHERENT", codes(validator.validate(inst, self.bundle)))

    def test_bound_power_uncertainty_cannot_be_dropped(self):
        record = find(self.bundle, "pwr.server")
        record["uncertainty_mw"] = 1
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_INCOHERENT",
                      codes(validator.validate(self.inst, self.bundle)))


class BoundarySemantics(unittest.TestCase):
    def setUp(self):
        self.bundle = load("mechanics_bundle.json")
        self.inst = load("activation_v3.json")

    def measured(self):
        """Promote the fixture to a MEASURED bundle for scope tests only."""
        bundle = copy.deepcopy(self.bundle)
        bundle["provenance"] = "MEASURED"
        for artifact in bundle["artifacts"]:
            artifact["provenance"] = "MEASURED"
        for record in bundle["power"]:
            record["instrument_kind"] = "COMPONENT_RAIL_METER"
            record["instrument"] = "synthetic_promoted_component_meter"
            reseal_record(record)
        reseal_bundle(bundle)
        inst = copy.deepcopy(self.inst)
        inst["evidence"]["scope"] = "MEASURED"
        rebind(inst, bundle)
        return inst, bundle

    def test_mechanics_bundle_can_never_support_a_physical_claim(self):
        cert = binder.solve_bound(self.inst, self.bundle)
        self.assertEqual(cert["evidence"]["energy_claim"], "NONE_MECHANICS_ONLY")

    def test_measured_instance_on_a_synthetic_bundle_is_caught(self):
        inst = copy.deepcopy(self.inst)
        inst["evidence"]["scope"] = "MEASURED"
        self.assertIn("E_PROVENANCE", codes(validator.validate(inst, self.bundle)))

    def test_measured_bundle_with_a_synthetic_artifact_is_caught(self):
        inst, bundle = self.measured()
        bundle["artifacts"][0]["provenance"] = "SYNTHETIC"
        reseal_bundle(bundle)
        rebind(inst, bundle)
        self.assertIn("E_PROVENANCE", codes(validator.validate(inst, bundle)))

    def test_nvml_presented_as_server_wall_is_caught(self):
        inst, bundle = self.measured()
        record = find(bundle, "pwr.server.zero")
        record["instrument_kind"] = "NVML"
        record["instrument"] = "nvidia-smi"
        reseal_record(record)
        reseal_bundle(bundle)
        rebind(inst, bundle)
        failures = validator.validate(inst, bundle)
        self.assertIn("E_SCOPE", codes(failures))
        self.assertTrue(any("cannot observe scope" in f for f in failures), failures)

    def test_nvml_presented_as_total_wall_is_caught(self):
        inst, bundle = self.measured()
        record = find(bundle, "pwr.server.zero")
        record["instrument_kind"] = "NVML"
        record["instrument"] = "nvml"
        record["scope"] = "TOTAL_WALL"
        reseal_record(record)
        reseal_bundle(bundle)
        rebind(inst, bundle)
        self.assertIn("E_SCOPE", codes(validator.validate(inst, bundle)))

    def test_gpu_board_power_cannot_authorize_a_physical_claim_in_e1(self):
        inst, bundle = self.measured()
        for rid in ("pwr.server.zero", "pwr.op15"):
            record = find(bundle, rid)
            record["scope"] = "GPU_BOARD"
            reseal_record(record)
        reseal_bundle(bundle)
        rebind(inst, bundle)
        self.assertIn("E_SCOPE", codes(validator.validate(inst, bundle)))
        with self.assertRaises(binder.EvidenceError):
            binder.solve_bound(inst, bundle)

    def test_server_wall_power_is_not_an_additive_device_term(self):
        inst, bundle = self.measured()
        self.assertIn("E_SCOPE", codes(validator.validate(inst, bundle)))
        with self.assertRaises(binder.EvidenceError):
            binder.solve_bound(inst, bundle)

    @staticmethod
    def promote(bundle, scope, boundaries=True):
        """Raise EVERY energy-contributing record to one scope."""
        for rid in ("pwr.server.zero", "pwr.op15"):
            record = find(bundle, rid)
            record["scope"] = scope
            reseal_record(record)
        if boundaries:
            for rid in ("bnd.a.p0", "bnd.a.p1", "bnd.a.c0", "bnd.a.c1"):
                record = find(bundle, rid)
                record["energy_scope"] = scope
                reseal_record(record)
        reseal_bundle(bundle)

    def test_total_wall_cannot_be_bound_as_additive_device_power(self):
        inst, bundle = self.measured()
        self.promote(bundle, "TOTAL_WALL")
        rebind(inst, bundle)
        failures = validator.validate(inst, bundle)
        self.assertIn("E_SCOPE", codes(failures))
        self.assertTrue(any("additive" in failure for failure in failures), failures)
        with self.assertRaises(binder.EvidenceError):
            binder.solve_bound(inst, bundle)

    def test_mixed_total_wall_and_gpu_board_terms_are_refused(self):
        inst, bundle = self.measured()
        self.promote(bundle, "TOTAL_WALL")
        record = find(bundle, "bnd.a.c0")
        record["energy_scope"] = "GPU_BOARD"
        reseal_record(record)
        reseal_bundle(bundle)
        rebind(inst, bundle)
        self.assertIn("E_SCOPE", codes(validator.validate(inst, bundle)))

    def test_unmeasured_zero_transfer_energy_is_refused_for_a_measured_claim(self):
        # The other door to "unknown encoded as zero": extra_energy_nj taken from a
        # boundary record that declares no boundary at all.
        inst, bundle = self.measured()
        record = find(bundle, "bnd.a.c0")
        record["energy_scope"] = "NONE"
        reseal_record(record)
        reseal_bundle(bundle)
        rebind(inst, bundle)
        failures = validator.validate(inst, bundle)
        self.assertIn("E_UNKNOWN_AS_ZERO", codes(failures))
        self.assertTrue(any("no boundary" in f for f in failures), failures)

    def test_one_total_wall_record_is_still_non_additive(self):
        inst, bundle = self.measured()
        record = find(bundle, "pwr.server.zero")
        record["scope"] = "TOTAL_WALL"
        reseal_record(record)
        reseal_bundle(bundle)
        rebind(inst, bundle)
        self.assertIn("E_SCOPE", codes(validator.validate(inst, bundle)))

    def test_unknown_phone_energy_encoded_as_zero_is_caught(self):
        inst, bundle = self.measured()
        record = find(bundle, "pwr.op15")
        record["status"] = "UNKNOWN"
        record["reason_code"] = "PHONE_ENERGY_PHYSICALLY_UNMEASURABLE"
        record["active_mw"] = 0
        reseal_record(record)
        reseal_bundle(bundle)
        inst["devices"]["OP15"]["active_mw"] = 0
        entry = next(b for b in inst["evidence"]["bindings"]
                     if b["target"] == "devices.OP15.active_mw")
        entry["value"] = 0
        rebind(inst, bundle)
        failures = validator.validate(inst, bundle)
        self.assertIn("E_UNKNOWN_AS_ZERO", codes(failures))
        self.assertTrue(any("never zero" in f for f in failures), failures)

    def test_unknown_phone_energy_cannot_be_bound_at_all(self):
        inst, bundle = self.measured()
        record = find(bundle, "pwr.op15")
        record["status"] = "UNKNOWN"
        record["reason_code"] = "PHONE_ENERGY_PHYSICALLY_UNMEASURABLE"
        reseal_record(record)
        reseal_bundle(bundle)
        rebind(inst, bundle)
        self.assertIn("E_STATUS", codes(validator.validate(inst, bundle)))

    def test_usb_vbus_double_counting_is_caught(self):
        inst, bundle = self.measured()
        record = find(bundle, "pwr.server.zero")
        record["included_rails"] = ["SERVER/CPU", "SERVER/DRAM",
                                    "SERVER/USB_VBUS"]
        record["excluded_rails"] = []
        reseal_record(record)
        reseal_bundle(bundle)
        rebind(inst, bundle)
        failures = validator.validate(inst, bundle)
        self.assertIn("E_DOUBLE_COUNT", codes(failures))
        self.assertTrue(any("twice" in f for f in failures), failures)

    def test_a_rail_both_included_and_excluded_is_caught(self):
        inst, bundle = self.measured()
        record = find(bundle, "pwr.op15")
        record["included_rails"] = ["OP15/SOC", "SERVER/USB_VBUS"]
        record["excluded_rails"] = ["SERVER/USB_VBUS"]
        reseal_record(record)
        reseal_bundle(bundle)
        rebind(inst, bundle)
        self.assertIn("E_DOUBLE_COUNT", codes(validator.validate(inst, bundle)))

    def test_unsynchronized_total_wall_is_caught(self):
        inst, bundle = self.measured()
        record = find(bundle, "pwr.server.zero")
        record["scope"] = "TOTAL_WALL"
        record["synchronization"] = "NONE"
        reseal_record(record)
        reseal_bundle(bundle)
        rebind(inst, bundle)
        self.assertIn("E_SCOPE", codes(validator.validate(inst, bundle)))

    def test_boundary_energy_without_a_scope_is_caught(self):
        bundle = copy.deepcopy(self.bundle)
        record = find(bundle, "bnd.a.c0")
        record["energy_nj"] = 5
        record["energy_scope"] = "NONE"
        reseal_record(record)
        reseal_bundle(bundle)
        self.assertIn("E_SCOPE", codes(validator.validate_bundle(bundle)))

    def test_certificate_claiming_more_than_the_evidence_supports_is_rejected(self):
        inst, bundle = self.inst, self.bundle
        cert = binder.solve_bound(inst, bundle)
        self.assertEqual(cert["evidence"]["energy_claim"], "NONE_MECHANICS_ONLY")
        cert["evidence"]["energy_claim"] = "SYSTEM_ENERGY_SAVING"
        cert["certificate_sha256"] = canon.digest(
            {k: v for k, v in cert.items() if k != "certificate_sha256"})
        failures = binder.check_bound(inst, bundle, cert)
        self.assertIn("E_SCOPE", codes(failures))
        self.assertTrue(any("supports at most" in f for f in failures), failures)


class FrozenEquations(unittest.TestCase):
    def test_delta_e_server_is_control_minus_qpim(self):
        self.assertEqual(boundary.delta_e_server(1000, 400), 600)
        self.assertEqual(
            boundary.phone_plus_external_break_even_budget(1000, 400), 600)

    def test_a_positive_server_delta_is_not_a_system_saving(self):
        label = boundary.relief_label(boundary.CLAIM_SERVER, 600, valid=True)
        self.assertEqual(label, "SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED")
        self.assertNotEqual(label, boundary.CLAIM_SYSTEM)

    def test_a_positive_gpu_board_delta_is_not_a_system_saving(self):
        self.assertEqual(boundary.relief_label(boundary.CLAIM_GPU_BOARD, 600, True),
                         "GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED")

    def test_a_non_positive_delta_is_relief_fail(self):
        self.assertEqual(boundary.relief_label(boundary.CLAIM_SERVER, 0, True),
                         "RELIEF_FAIL")
        self.assertEqual(boundary.relief_label(boundary.CLAIM_SERVER, -1, True),
                         "RELIEF_FAIL")

    def test_invalid_measurement_overrides_any_claim(self):
        self.assertEqual(boundary.relief_label(boundary.CLAIM_SYSTEM, 10**9, False),
                         "MEASUREMENT_INVALID")

    def test_work_slo_and_window_mismatches_are_caught(self):
        control = {"completed_work": 100,
                   "slo_outcomes": {"met": 100, "tardy": 0, "rejected": 0},
                   "window_us": 10,
                   "scope": "SERVER_WALL"}
        for field, value, code in (("completed_work", 99, "E_WORK_MISMATCH"),
                                   ("slo_outcomes",
                                    {"met": 99, "tardy": 1, "rejected": 0},
                                    "E_SLO_MISMATCH"),
                                   ("window_us", 11, "E_WINDOW_MISMATCH"),
                                   ("scope", "GPU_BOARD", "E_SCOPE")):
            treatment = dict(control)
            treatment[field] = value
            failures = boundary.validate_relief_comparison(control, treatment)
            self.assertIn(code, codes(failures), (field, failures))

    def test_an_identical_comparison_passes(self):
        control = {"completed_work": 100,
                   "slo_outcomes": {"met": 100, "tardy": 0, "rejected": 0},
                   "window_us": 10,
                   "scope": "SERVER_WALL"}
        self.assertEqual(boundary.validate_relief_comparison(control, dict(control)), [])

    def test_empty_comparisons_do_not_pass_by_none_equality(self):
        self.assertIn("E_SCHEMA",
                      codes(boundary.validate_relief_comparison({}, {})))

    def test_boolean_or_float_delta_is_not_an_energy_result(self):
        self.assertEqual(boundary.relief_label(boundary.CLAIM_SERVER, True, True),
                         "MEASUREMENT_INVALID")
        self.assertEqual(boundary.relief_label(boundary.CLAIM_SERVER, 1.0, True),
                         "MEASUREMENT_INVALID")
        self.assertEqual(boundary.relief_label(boundary.CLAIM_SERVER, 1, 1),
                         "MEASUREMENT_INVALID")
        self.assertEqual(boundary.relief_label(boundary.CLAIM_SYSTEM, 1, True),
                         "MEASUREMENT_INVALID")


class ProjectionSwap(unittest.TestCase):
    """A swapped v2 projection must not validate against a v3 certificate."""

    def test_certificate_for_a_different_instance_is_rejected(self):
        bundle = load("mechanics_bundle.json")
        inst = load("transition_v3.json")
        cert = binder.solve_bound(inst, bundle)
        other = load("activation_v3.json")
        failures = binder.check_bound(other, bundle, cert)
        self.assertIn("E_HASH", codes(failures))

    def test_certificate_naming_the_v2_projection_digest_is_rejected(self):
        bundle = load("mechanics_bundle.json")
        inst = load("transition_v3.json")
        cert = binder.solve_bound(inst, bundle)
        cert["instance_sha256"] = canon.digest(binder.project_v2(inst))
        cert["certificate_sha256"] = canon.digest(
            {k: v for k, v in cert.items() if k != "certificate_sha256"})
        failures = binder.check_bound(inst, bundle, cert)
        self.assertIn("E_HASH", codes(failures))
        self.assertTrue(any("projected" in f for f in failures), failures)

    def test_tampered_binding_digest_in_the_certificate_is_rejected(self):
        bundle = load("mechanics_bundle.json")
        inst = load("transition_v3.json")
        cert = binder.solve_bound(inst, bundle)
        cert["evidence"]["binding_sha256"] = "c" * 64
        cert["certificate_sha256"] = canon.digest(
            {k: v for k, v in cert.items() if k != "certificate_sha256"})
        self.assertIn("E_HASH", codes(binder.check_bound(inst, bundle, cert)))

    def test_certificate_body_edited_without_reseal_is_rejected(self):
        bundle = load("mechanics_bundle.json")
        inst = load("transition_v3.json")
        cert = binder.solve_bound(inst, bundle)
        cert["objective"][3] = 1
        self.assertIn("E_HASH", codes(binder.check_bound(inst, bundle, cert)))

    def test_a_suboptimal_but_feasible_v3_certificate_is_rejected(self):
        bundle = load("mechanics_bundle.json")
        inst = load("transition_v3.json")
        cert = binder.solve_bound(inst, bundle)
        # The earliest, non-merging placement: feasible, 162000000 nJ, not optimal.
        for action in cert["actions"]:
            if action["members"] == ["n0"]:
                action["start_us"], action["finish_us"] = 50, 150
        cert["energy"] = {"server_nj": 162000000, "phone_nj": 0,
                          "phone_by_device_nj": {}, "total_nj": 162000000,
                          "server_p0_intervals": [[0, 200], [950, 1150]]}
        cert["objective"] = [0, 0, -2, 162000000]
        cert["request_outcomes"] = [
            {"request_id": "r0", "terminal_finish_us": 150, "outcome": "MET",
             "lateness_us": 0},
            {"request_id": "r1", "terminal_finish_us": 1100, "outcome": "MET",
             "lateness_us": 0}]
        cert["certificate_sha256"] = canon.digest(
            {k: v for k, v in cert.items() if k != "certificate_sha256"})
        failures = binder.check_bound(inst, bundle, cert)
        self.assertTrue(any("SUBOPTIMAL" in f for f in failures), failures)


class RedTeamRegressions(unittest.TestCase):
    """One test per hole found by the adversarial audit of this gate.

    Every one of these PASSED validation before the audit. They are kept
    individually so a regression names the exact hole that reopened.
    """

    def setUp(self):
        self.bundle = load("mechanics_bundle.json")
        self.inst = load("transition_v3.json")
        self.act = load("activation_v3.json")

    def bind_to(self, inst, target, record_id):
        record = find(self.bundle, record_id)
        entry = next(b for b in inst["evidence"]["bindings"] if b["target"] == target)
        entry["record_id"] = record_id
        entry["record_sha256"] = record["record_sha256"]
        return inst

    # --- F1: float / bool type confusion defeats every eligibility gate --------
    def test_f1_float_observed_ppm_cannot_hide_a_failed_correctness(self):
        # 900000.0 == 900000 is True, so a float sails through every equality and
        # every ordering comparison unless the type itself is rejected.
        record = find(self.bundle, "corr.m0.server")
        record["observed_ppm"] = 900000.0
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_SCHEMA", codes(failures))
        self.assertTrue(any("float" in f for f in failures), failures)

    def test_f1_float_fallback_count_cannot_hide_cpu_fallback(self):
        record = find(self.bundle, "corr.m0.server")
        record["no_fallback_proof"]["fallback_ops_observed"] = 1.0
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_SCHEMA", codes(validator.validate(self.inst, self.bundle)))

    def test_f1_bool_fallback_count_cannot_hide_cpu_fallback(self):
        record = find(self.bundle, "corr.m0.server")
        record["no_fallback_proof"]["fallback_ops_observed"] = True
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_SCHEMA", codes(failures))
        self.assertTrue(any("boolean" in f for f in failures), failures)

    def test_f1_float_active_mw_cannot_hide_active_below_idle(self):
        record = find(self.bundle, "pwr.server")
        record["active_mw"] = 300000.0
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_SCHEMA", codes(validator.validate(self.inst, self.bundle)))

    def test_f1_float_in_a_binding_value_is_rejected(self):
        entry = self.inst["evidence"]["bindings"][0]
        entry["value"] = float(entry["value"])
        self.assertIn("E_SCHEMA", codes(validator.validate(self.inst, self.bundle)))

    def test_f1_the_json_schema_provably_cannot_catch_this(self):
        # Recorded as an executable fact, not a footnote: JSON Schema draft6+
        # defines `integer` as any number with a zero fractional part, so the
        # schema is NOT a second line of defence against a float. Proven against
        # /usr/bin/jsonschema, the validator this suite actually runs.
        if not pathlib.Path("/usr/bin/jsonschema").exists():
            self.skipTest("/usr/bin/jsonschema is not installed")
        schema = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        schema.write(json.dumps({"type": "integer"}))
        schema.close()
        results = {}
        for label, literal in (("int", "900000"), ("float_zero_frac", "900000.0"),
                               ("float_frac", "900000.5"), ("string", '"900000"')):
            doc = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            doc.write(literal)
            doc.close()
            results[label] = subprocess.run(
                ["/usr/bin/jsonschema", "-i", doc.name, schema.name],
                capture_output=True).returncode
        self.assertEqual(results["int"], 0)
        self.assertEqual(results["float_zero_frac"], 0,
                         "900000.0 validating as an integer is exactly why the "
                         "schema cannot be relied on for this")
        self.assertNotEqual(results["float_frac"], 0)
        self.assertNotEqual(results["string"], 0)
        # Only a real type check catches it, which is why canon.check_integers
        # exists and runs before any comparison.
        self.assertFalse(canon.is_int(900000.0))
        self.assertEqual(900000.0, 900000)

    # --- F8: MAX_INT was never enforced on a live path ------------------------
    def test_f8_overflow_in_a_record_is_rejected(self):
        record = find(self.bundle, "route.n0.server")
        record["latency"]["p95_us"] = 2 ** 60
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.inst["nodes"][0]["routes"]["SERVER"]["duration_us"] = 2 ** 60
        entry = next(b for b in self.inst["evidence"]["bindings"]
                     if b["target"] == "nodes.n0.routes.SERVER.duration_us")
        entry["value"] = 2 ** 60
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_SCHEMA", codes(failures))
        self.assertTrue(any("range" in f for f in failures), failures)

    # --- F9: a binding to an absent field validated via None == None ----------
    def test_f9_binding_to_an_absent_record_field_is_rejected(self):
        record = find(self.bundle, "bnd.n0")
        del record["energy_nj"]
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        entry = next(b for b in self.inst["evidence"]["bindings"]
                     if b["target"] == "nodes.n0.routes.SERVER.extra_energy_nj")
        entry["value"] = None
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_SCHEMA", codes(failures))
        self.assertTrue(any("required property" in f or "not of type" in f
                            for f in failures), failures)

    # --- F10: internal errors escaped as tracebacks ---------------------------
    def test_f10_non_digit_batch_size_is_a_refusal_not_a_crash(self):
        self.inst["batch_profiles"] = {"bk": {"1": 100}}
        self.inst["nodes"][0]["batch_key"] = "bk"
        route = find(self.bundle, "route.n0.server")
        self.inst["evidence"]["bindings"].append(
            {"target": "batch_profiles.bk.notanumber", "record_id": "route.n0.server",
             "record_sha256": route["record_sha256"], "field": "latency.p95_us",
             "value": 100})
        failures = validator.validate_safe(self.inst, self.bundle)
        self.assertTrue(failures)
        self.assertTrue(all(f.startswith("E_") for f in failures), failures)

    def test_f10_a_malformed_bundle_never_raises_out_of_validate_safe(self):
        for broken in ({}, {"schema_version": 3}, [], "x", None,
                       {"schema_version": 3, "bundle_id": "x", "provenance": "MEASURED",
                        "artifacts": [], "correctness": [], "routes": [],
                        "boundaries": [], "thermal": [], "power": [],
                        "bundle_sha256": "0" * 64}):
            failures = validator.validate_safe(self.inst, broken)
            self.assertTrue(failures, broken)
            self.assertTrue(all(f.startswith("E_") for f in failures), failures)

    # --- F5: per-field record selection built an unmeasured machine -----------
    def test_f5_server_power_fields_must_come_from_one_record(self):
        self.bind_to(self.inst, "server_power.transition_nj", "pwr.server.zero")
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_INCOHERENT", codes(failures))
        self.assertTrue(any("no record in the bundle describes" in f
                            for f in failures), failures)

    def test_f5_device_active_mw_must_match_the_server_power_record(self):
        self.bind_to(self.inst, "devices.SERVER.active_mw", "pwr.server.zero")
        self.assertIn("E_INCOHERENT",
                      codes(validator.validate(self.inst, self.bundle)))

    # --- F6: thermal state was evidenced but never constrained ----------------
    def test_f6_routes_from_two_thermal_conditions_cannot_share_a_schedule(self):
        cold = copy.deepcopy(find(self.bundle, "thr.server"))
        cold["record_id"] = "thr.server.cold"
        cold["thermal_state"] = "COLD"
        reseal_record(cold)
        self.bundle["thermal"].append(cold)
        route = find(self.bundle, "route.n0.server")
        route["thermal_id"] = "thr.server.cold"
        reseal_record(route)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_THERMAL", codes(failures))
        self.assertTrue(any("cherry-picking" in f for f in failures), failures)

    def test_f6_routes_from_two_backend_builds_cannot_share_a_schedule(self):
        artifact = copy.deepcopy(self.bundle["artifacts"][0])
        artifact["artifact_id"] = "art.server.trace.v2"
        artifact["backend_build"] = "other-build"
        self.bundle["artifacts"].append(artifact)
        route = find(self.bundle, "route.n0.server")
        route["backend_build"] = "other-build"
        route["artifacts"] = ["art.server.trace.v2"]
        reseal_record(route)
        cert = find(self.bundle, "corr.m0.server")
        cert["backend_build"] = "other-build"
        reseal_record(cert)
        reseal_bundle(self.bundle)
        rebind(self.inst, self.bundle)
        self.assertIn("E_STALE", codes(validator.validate(self.inst, self.bundle)))

    # --- F7: the activation bound bound any PASS route ------------------------
    def test_f7_activation_bound_requires_a_designated_capacity_probe(self):
        self.bind_to(self.inst, "activation_mem_bound_bytes", "route.n0.server")
        failures = validator.validate(self.inst, self.bundle)
        self.assertIn("E_IDENTITY", codes(failures))
        self.assertTrue(any("CAPACITY_PROBE" in f for f in failures), failures)

    # --- F3: boundary records had no identity at all --------------------------
    def test_f3_boundary_record_of_another_node_is_rejected(self):
        self.bind_to(self.inst, "nodes.n0.output_bytes", "bnd.n1")
        self.assertIn("E_IDENTITY",
                      codes(validator.validate(self.inst, self.bundle)))

    def test_f3_boundary_record_of_another_device_is_rejected(self):
        self.bind_to(self.act, "nodes.c0.routes.OP15.extra_energy_nj", "bnd.a.p0")
        self.assertIn("E_IDENTITY", codes(validator.validate(self.act, self.bundle)))

    def test_f3_a_gpu_link_boundary_cannot_back_a_phone_route(self):
        record = find(self.bundle, "bnd.a.c0")
        record["direction"] = "HOST_TO_GPU"
        record["transport_domain"] = "PCIE"
        reseal_record(record)
        reseal_bundle(self.bundle)
        rebind(self.act, self.bundle)
        failures = validator.validate(self.act, self.bundle)
        self.assertIn("E_IDENTITY", codes(failures))
        self.assertTrue(any("not evidence for another" in f for f in failures),
                        failures)


class MalformedInput(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                             encoding="ascii")
        handle.write(text)
        handle.close()
        return handle.name

    def test_duplicate_json_keys_are_rejected(self):
        path = self.write('{"a": 1, "a": 2}')
        with self.assertRaises(canon.DuplicateKeyError):
            canon.load_strict(path)

    def test_nan_is_rejected(self):
        path = self.write('{"a": NaN}')
        with self.assertRaises(ValueError):
            canon.load_strict(path)

    def test_infinity_is_rejected(self):
        path = self.write('{"a": Infinity}')
        with self.assertRaises(ValueError):
            canon.load_strict(path)

    def test_truncated_json_is_rejected(self):
        path = self.write('{"a": 1')
        with self.assertRaises(ValueError):
            canon.load_strict(path)

    def test_bool_is_not_an_integer(self):
        # NOTE: is_int(True) is False is necessary but NOT sufficient. Asserting
        # only this is what let F1 through: the guards were written
        # "if is_int(a) and is_int(b) and <bad>", so a wrong type SKIPPED the
        # check instead of failing it. The real proof that bools and floats are
        # rejected lives in RedTeamRegressions, which drives them through
        # validate() on live records. This test only pins the primitive.
        self.assertFalse(canon.is_int(True))
        with self.assertRaises(ValueError):
            canon.check_int(True, "flag")
        with self.assertRaises(ValueError):
            canon.check_integers({"n": True}, "r")

    def test_float_is_not_an_integer(self):
        with self.assertRaises(ValueError):
            canon.check_int(1.0, "value")
        with self.assertRaises(ValueError):
            canon.check_integers({"n": 1.0}, "r")

    def test_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            canon.check_int(2 ** 53, "value")
        with self.assertRaises(ValueError):
            canon.check_integers({"n": 2 ** 60}, "r")

    def test_check_integers_walks_nested_structures(self):
        canon.check_integers({"a": [{"b": [1, 2]}, {"c": "s"}, {"d": None}]}, "r")
        for bad in ({"a": [{"b": [1, 2.0]}]}, {"a": [{"b": [True]}]},
                    {"a": {"b": {"c": 2 ** 60}}}):
            with self.assertRaises(ValueError, msg=bad):
                canon.check_integers(bad, "r")

    def test_unsupported_bundle_version_is_rejected(self):
        bundle = load("mechanics_bundle.json")
        bundle["schema_version"] = 4
        self.assertIn("E_VERSION", codes(validator.validate_bundle(bundle)))

    def test_unsupported_record_version_is_rejected(self):
        bundle = load("mechanics_bundle.json")
        bundle["routes"][0]["record_version"] = 2
        reseal_bundle(bundle)
        self.assertIn("E_VERSION", codes(validator.validate_bundle(bundle)))

    def test_unsupported_instance_version_is_rejected(self):
        inst = load("transition_v3.json")
        inst["schema_version"] = 2
        self.assertIn("E_VERSION",
                      codes(validator.validate(inst, load("mechanics_bundle.json"))))

    def test_unknown_bundle_field_is_rejected(self):
        bundle = load("mechanics_bundle.json")
        bundle["extra"] = "x"
        self.assertIn("E_SCHEMA", codes(validator.validate_bundle(bundle)))

    def test_unknown_evidence_field_is_rejected(self):
        inst = load("transition_v3.json")
        inst["evidence"]["extra"] = "x"
        self.assertIn("E_SCHEMA",
                      codes(validator.validate(inst, load("mechanics_bundle.json"))))

    def test_empty_identifier_is_rejected(self):
        bundle = load("mechanics_bundle.json")
        bundle["routes"][0]["record_id"] = ""
        reseal_bundle(bundle)
        self.assertIn("E_SCHEMA", codes(validator.validate_bundle(bundle)))

    def test_dotted_node_id_is_rejected_as_ambiguous(self):
        inst = load("transition_v3.json")
        inst["nodes"][0]["id"] = "n.0"
        failures = validator.validate(inst, load("mechanics_bundle.json"))
        self.assertIn("E_SCHEMA", codes(failures))
        self.assertTrue(any("ambiguous" in f for f in failures), failures)

    def test_dotted_device_name_is_rejected_as_ambiguous(self):
        inst = load("activation_v3.json")
        inst["devices"]["OP.15"] = inst["devices"].pop("OP15")
        failures = validator.validate(inst, load("mechanics_bundle.json"))
        self.assertIn("E_SCHEMA", codes(failures))


if __name__ == "__main__":
    unittest.main()
