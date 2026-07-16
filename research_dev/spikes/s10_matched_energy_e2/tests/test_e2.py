#!/usr/bin/env python3
"""Adversarial tests for the S10-V0-R-E2 matched-timeline gate.

Every test here tries to manufacture a relief claim that the evidence does not
support. The gate passes only if all of them fail closed with a stable E_* code.

Each negative is non-vacuous: the positive control (the unmutated fixture) is
asserted to pass the same path, so a test cannot go green because the whole check
was deleted.
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import comparator  # noqa: E402
import e2_canon as canon  # noqa: E402
import integrator  # noqa: E402
import make_e2_fixtures as mk  # noqa: E402

FIX = ROOT / "fixtures"


def load(name):
    return canon.load_strict(str(FIX / name))


def codes(failures):
    return sorted({f.split(":")[0] for f in failures})


def reseal(record):
    record["record_sha256"] = canon.record_digest(record)
    return record


class Positive(unittest.TestCase):
    """The unmutated fixtures must pass every structural check.

    Without this, every negative below could pass vacuously.
    """

    def test_control_and_treatment_validate(self):
        for name in ("control_timeline.json", "treatment_timeline.json"):
            self.assertEqual(comparator.validate_timeline(load(name), FIX), [], name)

    def test_repetition_set_validates(self):
        self.assertEqual(
            comparator.validate_repetition_set(load("repetition_set.json")), [])

    def test_pair_matches(self):
        self.assertEqual(
            comparator.check_match(load("control_timeline.json"),
                                   load("treatment_timeline.json")), [])

    def test_fixtures_are_deterministic_across_processes(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "make_e2_fixtures.py"), "--check"],
            capture_output=True, text=True,
            env={"PYTHONHASHSEED": "31337", "PATH": "/usr/bin:/bin",
                 "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(proc.returncode, 0, proc.stderr)


class FrozenIntegration(unittest.TestCase):
    """The zero-order-hold rule is frozen and tested exactly (CONTRACT.md 3)."""

    def test_zero_order_hold_is_exact_by_hand(self):
        # 100 mW held 10 us, then 200 mW held 10 us -> 1000 + 2000 = 3000 nJ.
        samples = [[0, 100], [10, 200], [20, 999]]
        self.assertEqual(integrator.integrate(samples, 0, 20), 3000)

    def test_last_sample_contributes_no_interval(self):
        # The trailing 999 mW above must not be integrated: it only closes the
        # previous hold. If it were held forward the answer would be larger.
        samples = [[0, 100], [10, 200], [20, 10 ** 6]]
        self.assertEqual(integrator.integrate(samples, 0, 20), 3000)

    def test_window_clipping_is_exact(self):
        samples = [[0, 100], [10, 200], [20, 300], [30, 0]]
        # [5, 25] -> 100*5 + 200*10 + 300*5 = 500 + 2000 + 1500 = 4000
        self.assertEqual(integrator.integrate(samples, 5, 25), 4000)

    def test_synthetic_fixture_integral_matches_hand_arithmetic(self):
        control = load("control_timeline.json")
        self.assertEqual(control["energy_nj"], 300_000 * 20_000_000)
        treatment = load("treatment_timeline.json")
        self.assertEqual(treatment["energy_nj"], 240_000 * 20_000_000)

    def test_no_idle_baseline_is_subtracted(self):
        # Gross energy only. A constant 100 mW floor must appear in the integral.
        flat = [[0, 100], [1000, 100]]
        self.assertEqual(integrator.integrate(flat, 0, 1000), 100_000)


class DecisionArithmetic(unittest.TestCase):
    def make(self, c_energy, c_unc, t_energy, t_unc, scope="GPU_BOARD"):
        control = {"energy_nj": c_energy, "uncertainty_nj": c_unc}
        treatment = {"energy_nj": t_energy, "uncertainty_nj": t_unc}
        return comparator.decide(control, treatment, scope)

    def test_ten_percent_gate_is_exact_integer_cross_multiplication(self):
        # control_lower 1000, treatment_upper 900 -> exactly 10 percent: passes.
        self.assertTrue(self.make(1000, 0, 900, 0)["meets_ten_percent_gate"])
        # 901 is 9.9 percent: fails. No float rounding may rescue it.
        self.assertFalse(self.make(1000, 0, 901, 0)["meets_ten_percent_gate"])

    def test_uncertainty_is_never_dropped_from_the_decision(self):
        # A 15 percent nominal win that is inside the error bars is NOT relief.
        detail = self.make(1000, 100, 850, 100)
        self.assertEqual(detail["control_lower_nj"], 900)
        self.assertEqual(detail["treatment_upper_nj"], 950)
        self.assertFalse(detail["relief"])

    def test_a_win_under_ten_percent_is_relief_fail_not_a_partial_win(self):
        detail = self.make(1000, 0, 950, 0)
        self.assertGreater(detail["conservative_margin_nj"], 0)
        self.assertFalse(detail["meets_ten_percent_gate"])
        self.assertFalse(detail["relief"])

    def test_gpu_board_delta_is_not_a_server_break_even_budget(self):
        detail = self.make(1000, 0, 400, 0, "GPU_BOARD")
        self.assertEqual(detail["boundary_delta_nj"], 600)
        self.assertIsNone(detail["server_wall_delta_nj"])
        self.assertIsNone(detail["phone_plus_external_break_even_budget_nj"])

    def test_server_wall_delta_is_the_external_break_even_budget(self):
        detail = self.make(1000, 0, 400, 0, "SERVER_WALL")
        self.assertEqual(detail["boundary_delta_nj"], 600)
        self.assertEqual(detail["server_wall_delta_nj"], 600)
        self.assertEqual(detail["phone_plus_external_break_even_budget_nj"], 600)

    def test_a_treatment_that_uses_more_energy_is_negative(self):
        detail = self.make(1000, 0, 1500, 0)
        self.assertEqual(detail["boundary_delta_nj"], -500)
        self.assertFalse(detail["relief"])

    def test_no_floats_enter_the_decision(self):
        detail = self.make(10 ** 15, 10 ** 3, 10 ** 14, 10 ** 3)
        for key, value in detail.items():
            if key not in ("relief", "meets_ten_percent_gate",
                           "server_wall_delta_nj",
                           "phone_plus_external_break_even_budget_nj"):
                self.assertIs(type(value), int, key)

    def test_built_gpu_comparison_keeps_server_fields_null(self):
        record = comparator.build_comparison(
            load("control_timeline.json"), load("treatment_timeline.json"),
            FIX, load("repetition_set.json"))
        self.assertEqual(comparator.schema_errors("comparison", record), [])
        self.assertIsNone(record["server_wall_delta_nj"])
        self.assertIsNone(record["phone_plus_external_break_even_budget_nj"])

    def test_gpu_schema_rejects_a_fabricated_server_budget(self):
        record = comparator.build_comparison(
            load("control_timeline.json"), load("treatment_timeline.json"),
            FIX, load("repetition_set.json"))
        record["server_wall_delta_nj"] = record["boundary_delta_nj"]
        record["phone_plus_external_break_even_budget_nj"] = \
            record["boundary_delta_nj"]
        reseal(record)
        self.assertTrue(comparator.schema_errors("comparison", record))


class SyntheticEmitsNoPhysicalResult(unittest.TestCase):
    def test_a_synthetic_twenty_percent_win_is_measurement_invalid(self):
        control, treatment = load("control_timeline.json"), load("treatment_timeline.json")
        label, reason, detail, failures = comparator.compare(control, treatment, FIX)
        # The arithmetic genuinely clears the bar...
        self.assertTrue(detail["relief"])
        self.assertTrue(detail["meets_ten_percent_gate"])
        # ...and the label is still refused, because the evidence is synthetic.
        self.assertEqual(label, comparator.LABEL_INVALID)
        self.assertEqual(reason, "SYNTHETIC_NO_PHYSICAL_CLAIM")
        self.assertIn("E_PROVENANCE", codes(failures))

    def test_synthetic_artifact_cannot_be_promoted_by_relabelling(self):
        control, treatment = load("control_timeline.json"), load("treatment_timeline.json")
        for record in (control, treatment):
            record["provenance"] = "MEASURED"
            reseal(record)
        # The raw and execution artifacts still say SYNTHETIC, so the relabel is
        # rejected before any decision.
        label, reason, _detail, _f = comparator.compare(control, treatment, FIX)
        self.assertEqual(label, comparator.LABEL_INVALID)
        self.assertEqual(reason, "TIMELINE_INVALID")
        self.assertIn("E_RAW_BINDING", codes(_f))


class ScopeAndCapability(unittest.TestCase):
    def measured_gpu(self):
        """Promote the fixture to a real NVML GPU_BOARD pair for scope tests."""
        control, treatment = load("control_timeline.json"), load("treatment_timeline.json")
        for record in (control, treatment):
            record["provenance"] = "MEASURED"
            record["instrument_kind"] = "NVML_BOARD"
            record["instrument_identity"] = "nvml:GPU-45b611d6"
            reseal(record)
        return control, treatment

    def test_relabelled_nvml_pair_is_not_measurement_evidence(self):
        control, treatment = self.measured_gpu()
        label, reason, detail, failures = comparator.compare(control, treatment, FIX)
        self.assertEqual(label, comparator.LABEL_INVALID)
        self.assertEqual(reason, "TIMELINE_INVALID")
        self.assertIn("E_RAW_BINDING", codes(failures))
        self.assertEqual(detail, {})

    def test_nvml_relabelled_server_wall_is_rejected(self):
        control, treatment = self.measured_gpu()
        for record in (control, treatment):
            record["scope"] = "SERVER_WALL"
            record["board_uuids"] = []
            record["instrument_label"] = "total server wall power (honest)"
            reseal(record)
        failures = comparator.validate_timeline(control, FIX)
        self.assertIn("E_SCOPE", codes(failures))
        self.assertTrue(any("relabelling" in f for f in failures), failures)

    def test_rapl_can_claim_no_scope_at_all(self):
        control, _t = self.measured_gpu()
        control["instrument_kind"] = "RAPL_PACKAGE"
        reseal(control)
        failures = comparator.validate_timeline(control, FIX)
        self.assertIn("E_SCOPE", codes(failures))
        self.assertTrue(any("component counter" in f for f in failures), failures)

    def test_rapl_relabelled_server_wall_without_coverage_is_rejected(self):
        control, _t = self.measured_gpu()
        control["instrument_kind"] = "RAPL_PACKAGE"
        control["scope"] = "SERVER_WALL"
        control["board_uuids"] = []
        reseal(control)
        self.assertIn("E_SCOPE", codes(comparator.validate_timeline(control, FIX)))

    def test_server_wall_without_a_capability_record_is_rejected(self):
        control, treatment = self.measured_gpu()
        for record in (control, treatment):
            record["scope"] = "SERVER_WALL"
            record["instrument_kind"] = "EXTERNAL_WALL_METER"
            record["instrument_identity"] = "wall-meter-0"
            record["board_uuids"] = []
            record["included_rails"] = ["SERVER/WALL"]
            reseal(record)
        failures = comparator.check_wall_capability(None, control, FIX)
        self.assertIn("E_INCOMPLETE_WALL", codes(failures))

    def test_incomplete_wall_coverage_is_rejected(self):
        control, treatment = self.measured_gpu()
        for record in (control, treatment):
            record["scope"] = "SERVER_WALL"
            record["instrument_kind"] = "EXTERNAL_WALL_METER"
            record["instrument_identity"] = "synthetic-wall-meter-0"
            record["board_uuids"] = []
            reseal(record)
        cap = mk.wall_capability(complete=False)
        failures = comparator.check_wall_capability(cap, control, FIX)
        self.assertIn("E_INCOMPLETE_WALL", codes(failures))
        self.assertTrue(any("covers_storage" in f for f in failures), failures)

    def test_v1_wall_declaration_cannot_certify_measured_coverage(self):
        control, treatment = self.measured_gpu()
        for record in (control, treatment):
            record["scope"] = "SERVER_WALL"
            record["instrument_kind"] = "EXTERNAL_WALL_METER"
            record["instrument_identity"] = "synthetic-wall-meter-0"
            record["board_uuids"] = []
            reseal(record)
        cap = mk.wall_capability(complete=True, provenance="MEASURED")
        proof = mk.wall_coverage_proof(complete=True, provenance="MEASURED")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shutil.copytree(FIX, root, dirs_exist_ok=True)
            proof_text = json.dumps(proof, indent=2, sort_keys=True) + "\n"
            (root / cap["coverage_proof_artifact_path"]).write_text(
                proof_text, encoding="ascii")
            cap_failures = comparator.check_wall_capability(cap, control, root)
            self.assertIn("E_CAPABILITY_UNCERTIFIED", codes(cap_failures))
            label, reason, detail, failures = comparator.compare(
                control, treatment, root, wall_capability=cap)
            self.assertEqual(label, comparator.LABEL_INVALID)
            self.assertEqual(reason, "TIMELINE_INVALID")
            self.assertIn("E_RAW_BINDING", codes(failures))
            self.assertEqual(detail, {})

    def test_capability_uncertainty_floor_is_enforced(self):
        control, _treatment = self.measured_gpu()
        control["scope"] = "SERVER_WALL"
        control["instrument_kind"] = "EXTERNAL_WALL_METER"
        control["instrument_identity"] = "synthetic-wall-meter-0"
        control["board_uuids"] = []
        control["included_rails"] = ["SERVER/WALL"]
        reseal(control)
        cap = mk.wall_capability(
            uncertainty_floor_mw=5001, provenance="MEASURED")
        proof = mk.wall_coverage_proof(
            uncertainty_floor_mw=5001, provenance="MEASURED")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shutil.copytree(FIX, root, dirs_exist_ok=True)
            (root / cap["coverage_proof_artifact_path"]).write_text(
                json.dumps(proof, indent=2, sort_keys=True) + "\n",
                encoding="ascii")
            failures = comparator.check_wall_capability(cap, control, root)
        self.assertIn("E_UNCERTAINTY", codes(failures))

    def test_capability_declaration_needs_its_hashed_proof(self):
        control, _treatment = self.measured_gpu()
        control["scope"] = "SERVER_WALL"
        control["instrument_kind"] = "EXTERNAL_WALL_METER"
        control["instrument_identity"] = "synthetic-wall-meter-0"
        control["board_uuids"] = []
        control["included_rails"] = ["SERVER/WALL"]
        reseal(control)
        cap = mk.wall_capability(provenance="MEASURED")
        cap["coverage_proof_sha256"] = "0" * 64
        reseal(cap)
        self.assertIn(
            "E_ARTIFACT_HASH",
            codes(comparator.check_wall_capability(cap, control, FIX)))

    def test_two_board_uncertainty_floor_is_additive(self):
        control, _treatment = self.measured_gpu()
        second = "GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf"
        control["board_uuids"].append(second)
        control["uncertainty_nj"] = 5000 * 20_000_000
        reseal(control)
        failures = comparator.validate_timeline(control, FIX)
        self.assertIn("E_UNCERTAINTY", codes(failures))
        self.assertTrue(any("200000000000" in failure for failure in failures),
                        failures)

    def test_gpu_board_timeline_must_name_its_board(self):
        control, _t = self.measured_gpu()
        control["board_uuids"] = []
        reseal(control)
        failures = comparator.validate_timeline(control, FIX)
        self.assertIn("E_SCOPE", codes(failures))

    def test_different_boards_do_not_match(self):
        control, treatment = self.measured_gpu()
        treatment["board_uuids"] = ["GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf"]
        reseal(treatment)
        self.assertIn("E_SCOPE_MISMATCH",
                      codes(comparator.check_match(control, treatment)))


class MatchingMutations(unittest.TestCase):
    def setUp(self):
        self.control = load("control_timeline.json")
        self.treatment = load("treatment_timeline.json")

    def mutate(self, field, value, record="treatment"):
        target = self.treatment if record == "treatment" else self.control
        target[field] = value
        reseal(target)
        return comparator.check_match(self.control, self.treatment)

    def test_mismatched_workload_digest(self):
        self.assertIn("E_WORKLOAD_MISMATCH",
                      codes(self.mutate("workload_digest", canon.digest({"w": 2}))))

    def test_mismatched_trace_digest(self):
        self.assertIn("E_WORKLOAD_MISMATCH",
                      codes(self.mutate("trace_digest", canon.digest({"t": 2}))))

    def test_mismatched_slo_policy_digest(self):
        self.assertIn("E_SLO_MISMATCH",
                      codes(self.mutate("slo_policy_digest", canon.digest({"s": 2}))))

    def test_different_offered_work(self):
        self.assertIn("E_WORK_MISMATCH", codes(self.mutate("offered_work", 999)))

    def test_different_completed_work(self):
        self.assertIn("E_WORK_MISMATCH", codes(self.mutate("completed_work", 999)))

    def test_different_terminal_outcomes(self):
        failures = self.mutate("outcomes", {"met": 990, "tardy": 10, "rejected": 0,
                                            "canceled": 0})
        self.assertIn("E_SLO_MISMATCH", codes(failures))
        self.assertTrue(any("not relief" in f for f in failures), failures)

    def test_different_scope(self):
        self.assertIn("E_SCOPE_MISMATCH", codes(self.mutate("scope", "SERVER_WALL")))

    def test_different_instrument_kind(self):
        self.assertIn("E_SCOPE_MISMATCH",
                      codes(self.mutate("instrument_kind", "NVML_BOARD")))

    def test_different_rail_coverage(self):
        self.assertIn("E_SCOPE_MISMATCH",
                      codes(self.mutate("included_rails", ["SERVER/WALL", "EXTRA"])))

    def test_different_clock_epoch(self):
        failures = self.mutate("clock_epoch_id", "boot-other-1")
        self.assertIn("E_CLOCK_EPOCH", codes(failures))

    def test_different_device_identity(self):
        self.assertIn("E_DEVICE_MISMATCH",
                      codes(self.mutate("device_identity", "server-other")))

    def test_different_source_revision(self):
        self.assertIn("E_BUILD_MISMATCH",
                      codes(self.mutate("source_revision", "other-source")))

    def test_different_build_revision(self):
        self.assertIn("E_BUILD_MISMATCH",
                      codes(self.mutate("build_revision", "other-build")))

    def test_different_synchronization_method(self):
        self.assertIn("E_CLOCK_EPOCH",
                      codes(self.mutate("synchronization", "PTP")))

    def test_two_roles_cannot_alias_one_power_artifact(self):
        for field in ("raw_artifact_id", "raw_artifact_path",
                      "raw_artifact_sha256"):
            self.treatment[field] = self.control[field]
        reseal(self.treatment)
        self.assertIn("E_DUPLICATE_RUN",
                      codes(comparator.check_match(self.control, self.treatment)))

    def test_different_effective_window(self):
        self.treatment["window_end_us"] = 20_000_000 - 200_000
        reseal(self.treatment)
        self.assertIn("E_WINDOW_MISMATCH",
                      codes(comparator.check_match(self.control, self.treatment)))

    def test_window_within_tolerance_still_matches(self):
        # Positive control for the window gate.
        self.treatment["window_end_us"] = 20_000_000 - 10_000
        reseal(self.treatment)
        self.assertEqual(comparator.check_match(self.control, self.treatment), [])

    def test_treatment_and_control_swapped_after_the_fact(self):
        # Swapping roles to make the loser the "control" must not validate.
        self.control["role"] = "Q_PIM_TREATMENT"
        self.treatment["role"] = "OPTIMIZED_SERVER_ONLY_CONTROL"
        reseal(self.control)
        reseal(self.treatment)
        self.assertIn("E_ROLE",
                      codes(comparator.check_match(self.control, self.treatment)))

    def test_control_compared_against_itself(self):
        other = copy.deepcopy(self.control)
        other["timeline_id"] = "tl.control.copy"
        other["role"] = "Q_PIM_TREATMENT"
        reseal(other)
        failures = comparator.check_match(self.control, other)
        self.assertIn("E_SAME_POLICY", codes(failures))

    def test_the_same_run_used_as_both_sides(self):
        failures = comparator.check_match(self.control, self.control)
        self.assertIn("E_DUPLICATE_RUN", codes(failures))


class TimelineIntegrity(unittest.TestCase):
    def setUp(self):
        self.control = load("control_timeline.json")

    def check(self):
        return comparator.validate_timeline(self.control, FIX)

    def test_work_left_in_flight_is_rejected(self):
        self.control["outcomes"] = {"met": 900, "tardy": 0, "rejected": 0,
                                    "canceled": 0}
        self.control["completed_work"] = 900
        reseal(self.control)
        failures = self.check()
        self.assertIn("E_WORK_NOT_CLOSED", codes(failures))
        self.assertTrue(any("in flight" in f for f in failures), failures)

    def test_completed_work_inconsistent_with_outcomes(self):
        self.control["completed_work"] = 500
        reseal(self.control)
        self.assertIn("E_WORK_MISMATCH", codes(self.check()))

    def test_too_few_independent_updates(self):
        self.control["independent_updates"] = 57
        reseal(self.control)
        failures = self.check()
        self.assertIn("E_UPDATES", codes(failures))
        self.assertTrue(any("does not create information" in f for f in failures),
                        failures)

    def test_excessive_sample_gap(self):
        self.control["max_gap_us"] = 500_000
        reseal(self.control)
        self.assertIn("E_GAP", codes(self.check()))

    def test_window_shorter_than_the_minimum(self):
        self.control["window_end_us"] = self.control["window_start_us"] + 1000
        reseal(self.control)
        self.assertIn("E_WINDOW", codes(self.check()))

    def test_window_outside_the_execution_markers(self):
        self.control["window_end_us"] = self.control["marker_end_us"] + 1
        reseal(self.control)
        self.assertIn("E_WINDOW", codes(self.check()))

    def test_favorable_subwindow_inside_markers_is_rejected(self):
        self.control["window_start_us"] += 100_000
        self.control["window_end_us"] -= 100_000
        reseal(self.control)
        failures = self.check()
        self.assertIn("E_WINDOW", codes(failures))
        self.assertTrue(any("favorable subwindow" in failure
                            for failure in failures), failures)

    def test_execution_metadata_cannot_be_resealed_without_run_evidence(self):
        self.control["source_revision"] = "different-source"
        reseal(self.control)
        self.assertIn("E_EXECUTION_BINDING", codes(self.check()))

    def test_uncertainty_below_the_nvml_vendor_floor(self):
        self.control["provenance"] = "MEASURED"
        self.control["instrument_kind"] = "NVML_BOARD"
        self.control["uncertainty_nj"] = 0
        reseal(self.control)
        failures = self.check()
        self.assertIn("E_UNCERTAINTY", codes(failures))
        self.assertTrue(any("vendor" in f for f in failures), failures)

    def test_unsynchronized_timeline_is_rejected(self):
        self.control["synchronization"] = "NONE"
        reseal(self.control)
        self.assertIn("E_CLOCK_EPOCH", codes(self.check()))

    def test_record_edited_without_resealing(self):
        self.control["energy_nj"] = 1
        self.assertIn("E_DIGEST", codes(self.check()))

    def test_cli_cannot_emit_fake_empty_repetition_digests(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "comparison.json"
            proc = subprocess.run([
                sys.executable, str(ROOT / "src" / "comparator.py"),
                "--control", str(FIX / "control_timeline.json"),
                "--treatment", str(FIX / "treatment_timeline.json"),
                "--trusted-root", str(FIX), "--out", str(output), "--quiet",
            ], capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("E_PAIRS", proc.stderr)
            self.assertFalse(output.exists())

    def test_rails_both_included_and_excluded(self):
        self.control["excluded_rails"] = list(self.control["included_rails"])
        reseal(self.control)
        self.assertIn("E_SCOPE", codes(self.check()))

    def test_system_energy_saving_injected_anywhere_is_caught(self):
        self.control["reason_code"] = "SYSTEM_ENERGY_SAVING"
        reseal(self.control)
        failures = self.check()
        self.assertIn("E_SYSTEM_CLAIM", codes(failures))

    def test_system_energy_saving_is_not_expressible_in_any_schema(self):
        # The strongest form: the forbidden label is absent from every enum, so it
        # cannot be emitted even by a bug.
        for name in ("matched_comparison.v1.schema.json",
                     "repetition_set.v1.schema.json",
                     "realized_timeline.v1.schema.json"):
            text = (ROOT / "schemas" / name).read_text(encoding="ascii")
            self.assertNotIn("SYSTEM_ENERGY_SAVING", text, name)
        self.assertNotIn(comparator.FORBIDDEN_LABEL, comparator.ALLOWED_LABELS)

    def test_build_comparison_cannot_be_handed_a_label(self):
        # The label is DERIVED from the evidence, never accepted. An earlier
        # revision took `label` and only checked membership in the allowed set,
        # so importing this module stamped a digest-valid, schema-valid
        # SERVER_RELIEF_PASS onto junk. The signature itself is now the guard.
        import inspect
        params = list(inspect.signature(comparator.build_comparison).parameters)
        self.assertNotIn("label", params)
        self.assertNotIn("reason", params)
        self.assertNotIn("detail", params)
        self.assertEqual(params[:3], ["control", "treatment", "trusted_root"])

    def test_build_comparison_cannot_launder_a_relief_label_onto_junk(self):
        # The red team's exploit, verbatim: a treatment that burned 1e15 nJ MORE,
        # on a SERVER_WALL scope this host cannot instrument, with no capability
        # record. It must be impossible to obtain a sealed relief record for it.
        control, treatment = load("control_timeline.json"), load("treatment_timeline.json")
        treatment["energy_nj"] = control["energy_nj"] + 10 ** 15
        for record in (control, treatment):
            record["provenance"] = "MEASURED"
            record["instrument_kind"] = "EXTERNAL_WALL_METER"
            record["scope"] = "SERVER_WALL"
            record["board_uuids"] = []
            reseal(record)
        with self.assertRaises(comparator.ComparisonError) as ctx:
            comparator.build_comparison(
                control, treatment, FIX, load("repetition_set.json"))
        self.assertIn(ctx.exception.code,
                      {"E_BINDING", "E_PAIRS", "TIMELINE_INVALID"})

    def test_a_failed_timeline_cannot_reach_the_decision(self):
        for status in ("FAILED", "INELIGIBLE"):
            control = load("control_timeline.json")
            control["status"] = status
            control["reason_code"] = "RUN_CRASHED"
            reseal(control)
            failures = comparator.validate_timeline(control, FIX)
            self.assertIn("E_PAIRS", codes(failures), status)
            self.assertTrue(any("cannot support a comparison" in f
                                for f in failures), failures)

    def test_a_split_label_is_NOT_caught_by_the_scan_and_need_not_be(self):
        # Honest limit, recorded as a test rather than a footnote. Canonical JSON
        # sorts keys, so two free-form fields are not adjacent, and NO contiguous
        # substring scan can catch a phrase split across arbitrary fields. That is
        # fine: the scan is defence in depth. What actually blocks the claim is
        # that `result_label` is a closed enum in both the schema and
        # ALLOWED_LABELS, so no combination of free-form strings can BE a label.
        split = {"first": "SYSTEM_ENERGY", "second": "_SAVING"}
        self.assertNotIn(comparator.FORBIDDEN_NEEDLE,
                         comparator._normalized_text(split))
        # ...and the thing that matters is still unreachable:
        self.assertNotIn(comparator.FORBIDDEN_LABEL, comparator.ALLOWED_LABELS)
        schema = json.loads((ROOT / "schemas"
                             / "matched_comparison.v1.schema.json").read_text())
        self.assertEqual(
            schema["properties"]["result_label"]["const"],
            comparator.LABEL_INVALID)

    def test_lowercase_forbidden_label_is_caught(self):
        control = load("control_timeline.json")
        control["reason_code"] = "system_energy_saving"
        reseal(control)
        self.assertIn("E_SYSTEM_CLAIM",
                      codes(comparator.validate_timeline(control, FIX)))


class SampleMutations(unittest.TestCase):
    def test_nonmonotonic_timestamps(self):
        with self.assertRaises(integrator.TimelineError) as ctx:
            integrator.normalize_samples([[0, 10], [20, 10], [15, 10]])
        self.assertEqual(ctx.exception.code, "E_NONMONOTONIC")

    def test_duplicate_timestamps(self):
        with self.assertRaises(integrator.TimelineError) as ctx:
            integrator.normalize_samples([[0, 10], [10, 20], [10, 30]])
        self.assertEqual(ctx.exception.code, "E_NONMONOTONIC")

    def test_float_power_is_rejected_before_any_comparison(self):
        # 240000.0 == 240000 is True in Python, so a float would satisfy every
        # equality and ordering test. Only the type gate catches it.
        with self.assertRaises(integrator.TimelineError) as ctx:
            integrator.normalize_samples([[0, 240000.0], [1000, 100]])
        self.assertEqual(ctx.exception.code, "E_SCHEMA")
        self.assertIn("float", str(ctx.exception))

    def test_float_timestamp_is_rejected(self):
        with self.assertRaises(integrator.TimelineError):
            integrator.normalize_samples([[0.0, 100], [1000, 100]])

    def test_bool_power_is_rejected(self):
        with self.assertRaises(integrator.TimelineError) as ctx:
            integrator.normalize_samples([[0, True], [1000, 100]])
        self.assertIn("boolean", str(ctx.exception))

    def test_negative_power_is_rejected(self):
        with self.assertRaises(integrator.TimelineError):
            integrator.normalize_samples([[0, -1], [1000, 100]])

    def test_negative_timestamp_is_rejected(self):
        with self.assertRaises(integrator.TimelineError):
            integrator.normalize_samples([[-1, 100], [1000, 100]])

    def test_overflow_power_is_rejected(self):
        with self.assertRaises(integrator.TimelineError):
            integrator.normalize_samples([[0, 2 ** 60], [1000, 100]])

    def test_unbracketed_window_start(self):
        with self.assertRaises(integrator.TimelineError) as ctx:
            integrator.integrate([[100, 10], [2_000_000, 10]], 0, 2_000_000)
        self.assertEqual(ctx.exception.code, "E_UNBRACKETED")

    def test_unbracketed_window_end(self):
        with self.assertRaises(integrator.TimelineError) as ctx:
            integrator.integrate([[0, 10], [2_000_000, 10]], 0, 3_000_000)
        self.assertEqual(ctx.exception.code, "E_UNBRACKETED")

    def test_missing_marker_leaves_no_window(self):
        with self.assertRaises(integrator.TimelineError) as ctx:
            integrator.integrate([[0, 10], [2_000_000, 10]], 5, 5)
        self.assertEqual(ctx.exception.code, "E_WINDOW")

    def test_too_few_samples(self):
        with self.assertRaises(integrator.TimelineError):
            integrator.normalize_samples([[0, 10]])

    def test_recomputation_catches_a_forged_energy(self):
        # The artifact bytes and their hash are untouched and correct; only the
        # record lies about what they integrate to. Recomputation is the ONLY
        # thing that catches this, which is why it is mandatory rather than
        # opt-in.
        control = load("control_timeline.json")
        control["energy_nj"] = 1
        reseal(control)
        failures = comparator.validate_timeline(control, FIX)
        self.assertIn("E_SCHEMA", codes(failures))
        self.assertTrue(any("zero-order-hold integral" in f for f in failures),
                        failures)

    def test_recomputation_accepts_the_honest_record(self):
        self.assertEqual(
            comparator.validate_timeline(load("control_timeline.json"), FIX), [])

    def test_recomputation_catches_a_forged_update_count(self):
        control = load("control_timeline.json")
        control["independent_updates"] = 100000
        reseal(control)
        self.assertIn("E_UPDATES",
                      codes(comparator.validate_timeline(control, FIX)))

    def test_recomputation_catches_a_forged_gap(self):
        control = load("control_timeline.json")
        control["max_gap_us"] = 1
        reseal(control)
        self.assertIn("E_GAP", codes(comparator.validate_timeline(control, FIX)))

    def test_recomputation_catches_a_forged_sample_count(self):
        control = load("control_timeline.json")
        control["sample_count"] = 999
        reseal(control)
        self.assertIn("E_SCHEMA", codes(comparator.validate_timeline(control, FIX)))

    def test_a_forged_energy_cannot_reach_a_physical_label(self):
        # The end-to-end form of the same attack. Synthetic evidence is already
        # diagnostic-only; forging its energy must fail even before that boundary.
        control, treatment = load("control_timeline.json"), load("treatment_timeline.json")
        # Sanity: the honest pair reaches only the synthetic diagnostic result.
        label, reason, _d, failures = comparator.compare(control, treatment, FIX)
        self.assertEqual(label, comparator.LABEL_INVALID)
        self.assertEqual(reason, "SYNTHETIC_NO_PHYSICAL_CLAIM")
        self.assertIn("E_PROVENANCE", codes(failures))
        # Now forge the treatment energy down to 1 nJ, leaving the artifact intact.
        treatment["energy_nj"] = 1
        reseal(treatment)
        label, reason, _d, failures = comparator.compare(control, treatment, FIX)
        self.assertEqual(label, comparator.LABEL_INVALID)
        self.assertEqual(reason, "TIMELINE_INVALID")
        self.assertTrue(any("zero-order-hold integral" in f for f in failures),
                        failures)

    def test_compare_has_no_way_to_skip_recomputation(self):
        # A `samples=None` opt-in was the original defect. Assert the parameter is
        # gone, so recomputation cannot be silently declined by a caller.
        import inspect
        params = set(inspect.signature(comparator.compare).parameters)
        self.assertEqual(params,
                         {"control", "treatment", "trusted_root", "wall_capability"})
        tl_params = set(inspect.signature(comparator.validate_timeline).parameters)
        self.assertEqual(tl_params, {"record", "trusted_root"})


class WindowPadding(unittest.TestCase):
    """F2: quality must be measured over the interval the energy is paid from.

    The red team padded busy samples OUTSIDE the paid window. They contribute zero
    energy (the integral clips to the window) but, when changes were counted over
    the whole artifact, they bought unlimited `independent_updates`. It admitted
    the exact real trace CONTRACT.md section 7 says is rejected.
    """

    def build(self, tmp, pad):
        """A window holding 3 changes, optionally padded with 200 more outside."""
        root = pathlib.Path(tmp)
        (root / "raw").mkdir()
        (root / "execution").mkdir()
        window_start, window_end = 5_000_000, 7_000_000
        samples, pstates = [], []
        # Padding BEFORE the window: many changes, zero paid energy.
        if pad:
            for i in range(100):
                samples.append([i * 40_000, 1000 + (i % 2)])
                pstates.append("P0")
        # The paid window: only 3 changes, far below the gate.
        samples.append([window_start, 1000])
        samples.append([window_start + 1_000_000, 1001])
        samples.append([window_end, 1002])
        pstates.extend(["P0"] * 3)
        # Padding AFTER the window.
        if pad:
            for i in range(100):
                samples.append([window_end + 40_000 * (i + 1), 2000 + (i % 2)])
                pstates.append("P0")
        record = load("control_timeline.json")
        record["provenance"] = "MEASURED"
        record["instrument_kind"] = "NVML_BOARD"
        record["raw_artifact_id"] = "padded.json"
        record["raw_artifact_path"] = "raw/padded.json"
        record["marker_start_us"] = window_start
        record["marker_end_us"] = window_end
        record["window_start_us"] = window_start
        record["window_end_us"] = window_end
        normalized = integrator.normalize_samples(samples)
        record["sample_count"] = len(normalized)
        record["independent_updates"], record["max_gap_us"] = \
            integrator.window_quality(normalized, window_start, window_end)
        record["energy_nj"] = integrator.integrate(normalized, window_start,
                                                   window_end)
        record["paid_payload_sha256"] = comparator.paid_payload_digest(
            normalized, pstates, window_start, window_end)
        record["uncertainty_nj"] = 5000 * (window_end - window_start)
        raw = {"schema": "e2.raw.v2", "pstates": pstates, "samples": samples}
        raw.update({field: record[field] for field in comparator.RAW_BOUND_FIELDS})
        text = json.dumps(raw, indent=2, sort_keys=True) + "\n"
        (root / "raw" / "padded.json").write_text(text, encoding="ascii")
        record["raw_artifact_sha256"] = canon.sha256_bytes(text.encode("ascii"))
        execution = {"schema": "e2.execution.v2"}
        execution.update({field: record[field]
                          for field in comparator.EXECUTION_BOUND_FIELDS})
        execution_text = json.dumps(execution, indent=2, sort_keys=True) + "\n"
        (root / record["execution_artifact_path"]).write_text(
            execution_text, encoding="ascii")
        record["execution_artifact_sha256"] = canon.sha256_bytes(
            execution_text.encode("ascii"))
        reseal(record)
        return record, root

    def test_padding_outside_the_window_buys_no_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, root = self.build(tmp, pad=True)
            # Honest accounting of the padded artifact still sees only the window.
            self.assertEqual(record["independent_updates"], 1)
            failures = comparator.validate_timeline(record, root)
            self.assertIn("E_UPDATES", codes(failures))
            self.assertTrue(any("buy no quality" in f or "does not create" in f
                                for f in failures), failures)

    def test_declaring_the_whole_artifact_update_count_is_caught(self):
        # The actual exploit: claim the GLOBAL change count for a sparse window.
        with tempfile.TemporaryDirectory() as tmp:
            record, root = self.build(tmp, pad=True)
            data = (root / "raw" / "padded.json").read_text(encoding="ascii")
            samples = json.loads(data)["samples"]
            normalized = integrator.normalize_samples(samples)
            record["independent_updates"] = integrator.independent_updates(normalized)
            self.assertGreaterEqual(record["independent_updates"],
                                    integrator.MIN_INDEPENDENT_UPDATES)
            reseal(record)
            failures = comparator.validate_timeline(record, root)
            self.assertIn("E_UPDATES", codes(failures))

    def test_window_slice_counts_only_in_window_changes(self):
        samples = [[0, 1], [10, 2], [20, 3], [30, 4], [40, 5]]
        # The value reported exactly at 30 applies after the paid interval.
        self.assertEqual(integrator.window_quality(samples, 20, 30), (0, 10))
        # window_slice remains a bracketing primitive, not the quality rule.
        self.assertEqual(
            integrator.independent_updates(
                integrator.window_slice(samples, 20, 30)), 1)
        self.assertEqual(integrator.independent_updates(samples), 4)


class ArtifactIntegrity(unittest.TestCase):
    def setUp(self):
        self.control = load("control_timeline.json")

    def test_missing_artifact(self):
        self.control["raw_artifact_path"] = "raw/nope.json"
        reseal(self.control)
        self.assertIn("E_ARTIFACT_MISSING",
                      codes(comparator.validate_timeline(self.control, FIX)))

    def test_modified_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "raw").mkdir()
            original = (FIX / "raw" / "control.json").read_text(encoding="ascii")
            (root / "raw" / "control.json").write_text(
                original.replace("300001", "100001"), encoding="ascii")
            failures = comparator.validate_timeline(self.control, root)
            self.assertIn("E_ARTIFACT_HASH", codes(failures))

    def test_truncated_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "raw").mkdir()
            (root / "raw" / "control.json").write_text('{"samples": [', encoding="ascii")
            self.assertIn("E_ARTIFACT_HASH",
                          codes(comparator.validate_timeline(self.control, root)))

    def test_path_traversal(self):
        self.control["raw_artifact_path"] = "../../../../etc/passwd"
        reseal(self.control)
        failures = comparator.validate_timeline(self.control, FIX)
        self.assertIn("E_ARTIFACT_PATH", codes(failures))
        self.assertTrue(any("upward" in f for f in failures), failures)

    def test_absolute_path(self):
        self.control["raw_artifact_path"] = "/etc/passwd"
        reseal(self.control)
        self.assertIn("E_ARTIFACT_PATH",
                      codes(comparator.validate_timeline(self.control, FIX)))

    def test_symlink_is_refused_even_when_it_points_inside(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "raw").mkdir()
            real = root / "raw" / "real.json"
            real.write_text((FIX / "raw" / "control.json").read_text(encoding="ascii"),
                            encoding="ascii")
            link = root / "raw" / "control.json"
            os.symlink(real, link)
            failures = comparator.validate_timeline(self.control, root)
            self.assertIn("E_ARTIFACT_PATH", codes(failures))
            self.assertTrue(any("symlink" in f for f in failures), failures)


class ArtifactReadOnce(unittest.TestCase):
    """F3: the hash and the integral must examine the SAME bytes.

    An earlier revision hashed the path in one open, then re-opened the same path
    to parse it. Two opens can see two files; the red team won that race 74/400
    with no privileges, producing a record whose declared energy described bytes
    other than the ones it pinned. A race is not deterministically testable, so
    the property is pinned structurally: there is exactly one read, and the parser
    cannot take a path.
    """

    def test_the_verified_bytes_are_returned_for_parsing(self):
        path, data = integrator.read_verified_artifact(
            "raw/control.json",
            load("control_timeline.json")["raw_artifact_sha256"], FIX)
        self.assertIsInstance(data, bytes)
        self.assertEqual(canon.sha256_bytes(data),
                         load("control_timeline.json")["raw_artifact_sha256"])
        # The parser consumes those bytes, and cannot be handed a path instead.
        raw = comparator.parse_raw_artifact(data)
        self.assertEqual(len(raw["samples"]), len(raw["pstates"]))
        with self.assertRaises(AttributeError):
            comparator.parse_raw_artifact(str(path))

    def test_the_old_two_open_helper_is_gone(self):
        # verify_artifact() returned only a path, forcing the caller to re-open.
        self.assertFalse(hasattr(integrator, "verify_artifact"),
                         "the two-open helper must not come back")

    def test_a_swapped_file_after_hashing_cannot_change_the_integral(self):
        # Simulate the race deterministically: hash the real bytes, then replace
        # the file entirely. The integral must still come from the hashed buffer.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "raw").mkdir()
            honest = (FIX / "raw" / "control.json").read_text(encoding="ascii")
            target = root / "raw" / "control.json"
            target.write_text(honest, encoding="ascii")
            _path, data = integrator.read_verified_artifact(
                "raw/control.json",
                canon.sha256_bytes(honest.encode("ascii")), root)
            # The attacker swaps the file now, after the hash.
            target.write_text(json.dumps({"schema": "e2.raw.v1", "pstates": ["P0"],
                                          "samples": [[0, 1]]}), encoding="ascii")
            raw = comparator.parse_raw_artifact(data)
            self.assertEqual(len(raw["samples"]), 201,
                             "the parse used the swapped file, not the hashed bytes")


class RepetitionSetMutations(unittest.TestCase):
    def setUp(self):
        self.record = load("repetition_set.json")

    def check(self):
        return comparator.validate_repetition_set(self.record)

    def test_dropped_unfavorable_repetition(self):
        # attempted_pairs still says 8, but only 7 are listed: the classic
        # quiet removal of a run that did not help.
        self.record["pairs"] = self.record["pairs"][:-1]
        reseal(self.record)
        failures = self.check()
        self.assertIn("E_PAIRS", codes(failures))
        self.assertTrue(any("dropped unfavorable run is a diff" in f
                            for f in failures), failures)

    def test_too_few_pairs(self):
        self.record["pairs"] = self.record["pairs"][:4]
        self.record["attempted_pairs"] = 4
        reseal(self.record)
        self.assertIn("E_PAIRS", codes(self.check()))

    def test_failed_pair_invalidates_the_set(self):
        self.record["pairs"][0]["status"] = "FAILED"
        self.record["pairs"][0]["reason_code"] = "RUN_CRASHED"
        reseal(self.record)
        failures = self.check()
        self.assertIn("E_PAIRS", codes(failures))
        self.assertTrue(any("cannot be excluded" in f for f in failures), failures)

    def test_failed_pair_cannot_be_hidden_behind_eight_ok_pairs(self):
        failed = copy.deepcopy(self.record["pairs"][-1])
        failed["pair_index"] = 8
        failed["control_timeline_id"] = "tl.control.failed"
        failed["treatment_timeline_id"] = "tl.treatment.failed"
        failed["first_executed_role"] = "OPTIMIZED_SERVER_ONLY_CONTROL"
        failed["status"] = "FAILED"
        failed["reason_code"] = "RUN_CRASHED"
        self.record["pairs"].append(failed)
        self.record["attempted_pairs"] = 9
        self.record["declared_order"].extend([
            "OPTIMIZED_SERVER_ONLY_CONTROL", "Q_PIM_TREATMENT"])
        self.record["order_digest"] = canon.digest(self.record["declared_order"])
        reseal(self.record)
        failures = self.check()
        self.assertIn("E_PAIRS", codes(failures))
        self.assertTrue(any("cannot be excluded" in f for f in failures), failures)

    def test_duplicate_run_reused_across_pairs(self):
        self.record["pairs"][1]["control_timeline_id"] = \
            self.record["pairs"][0]["control_timeline_id"]
        reseal(self.record)
        failures = self.check()
        self.assertIn("E_DUPLICATE_RUN", codes(failures))
        self.assertTrue(any("fakes" in f for f in failures), failures)

    def test_pair_indices_must_be_contiguous_from_zero(self):
        self.record["pairs"][0]["pair_index"] = 10
        reseal(self.record)
        self.assertIn("E_PAIRS", codes(self.check()))

    def test_warmup_cannot_be_reused_as_a_measured_run(self):
        pair = self.record["pairs"][0]
        self.record["warmup_timelines"] = [{
            "timeline_id": pair["control_timeline_id"],
            "record_sha256": pair["control_record_sha256"],
        }]
        reseal(self.record)
        self.assertIn("E_DUPLICATE_RUN", codes(self.check()))

    def test_record_digest_cannot_be_reused_under_a_new_timeline_id(self):
        self.record["pairs"][1]["control_record_sha256"] = \
            self.record["pairs"][0]["control_record_sha256"]
        reseal(self.record)
        self.assertIn("E_DUPLICATE_RUN", codes(self.check()))

    def test_warmup_digest_cannot_alias_a_measured_run_under_a_new_id(self):
        self.record["warmup_timelines"] = [{
            "timeline_id": "tl.warmup.alias",
            "record_sha256": self.record["pairs"][0]["control_record_sha256"],
        }]
        reseal(self.record)
        self.assertIn("E_DUPLICATE_RUN", codes(self.check()))

    def test_scope_and_instrument_pair_is_typed(self):
        self.record["scope"] = "SERVER_WALL"
        self.record["instrument_kind"] = "NVML_BOARD"
        reseal(self.record)
        self.assertIn("E_SCOPE", codes(self.check()))

    def test_control_and_treatment_bind_distinct_record_digests(self):
        self.record["pairs"][0]["treatment_record_sha256"] = \
            self.record["pairs"][0]["control_record_sha256"]
        reseal(self.record)
        self.assertIn("E_DUPLICATE_RUN", codes(self.check()))

    def test_unbalanced_execution_order(self):
        for pair in self.record["pairs"]:
            pair["first_executed_role"] = "Q_PIM_TREATMENT"
        reseal(self.record)
        failures = self.check()
        self.assertIn("E_ORDER", codes(failures))
        self.assertTrue(any("unbalanced" in f for f in failures), failures)

    def test_order_digest_not_covering_the_declared_order(self):
        self.record["declared_order"][0] = "Q_PIM_TREATMENT"
        reseal(self.record)
        failures = self.check()
        self.assertIn("E_ORDER", codes(failures))

    def test_repetition_set_cannot_self_declare_a_result(self):
        self.record["aggregate_result_label"] = "RELIEF_FAIL"
        reseal(self.record)
        self.assertIn("E_SCHEMA", codes(self.check()))

    def test_extra_unexecuted_order_entries_are_rejected(self):
        self.record["declared_order"].extend([
            "OPTIMIZED_SERVER_ONLY_CONTROL", "Q_PIM_TREATMENT"])
        self.record["order_digest"] = canon.digest(self.record["declared_order"])
        reseal(self.record)
        failures = self.check()
        self.assertIn("E_ORDER", codes(failures))
        self.assertTrue(any("exactly" in f for f in failures), failures)


class PostReviewRegressions(unittest.TestCase):
    def raw_document(self):
        return canon.load_strict(str(FIX / "raw" / "control.json"))

    def parse_mutated_raw(self, value):
        raw = self.raw_document()
        raw["pstates"][0] = value
        return comparator.parse_raw_artifact(canon.canonical(raw))

    def test_pstate_null_is_rejected(self):
        with self.assertRaises(integrator.TimelineError) as ctx:
            self.parse_mutated_raw(None)
        self.assertEqual(ctx.exception.code, "E_SCHEMA")

    def test_pstate_container_is_rejected_without_typeerror(self):
        with self.assertRaises(integrator.TimelineError) as ctx:
            self.parse_mutated_raw(["P0"])
        self.assertEqual(ctx.exception.code, "E_SCHEMA")

    def test_pstate_empty_string_is_rejected(self):
        with self.assertRaises(integrator.TimelineError) as ctx:
            self.parse_mutated_raw("")
        self.assertEqual(ctx.exception.code, "E_SCHEMA")

    def test_end_marker_status_does_not_change_the_paid_window(self):
        record = load("control_timeline.json")
        raw = self.raw_document()
        raw["pstates"][-1] = "P8"
        failures = comparator._verify_recomputation(
            record, raw["samples"], raw["pstates"])
        self.assertNotIn("E_STATUS_CHANGE", codes(failures))
        self.assertNotIn("E_RAW_BINDING", codes(failures))

    def test_interior_status_change_is_rejected(self):
        record = load("control_timeline.json")
        raw = self.raw_document()
        raw["pstates"][10] = "P8"
        failures = comparator._verify_recomputation(
            record, raw["samples"], raw["pstates"])
        self.assertIn("E_STATUS_CHANGE", codes(failures))

    def test_start_bracket_age_counts_toward_max_gap(self):
        samples = [[0, 10], [10_100_000, 11], [10_200_000, 12]]
        self.assertEqual(
            integrator.window_quality(samples, 10_000_000, 10_200_000),
            (1, 10_100_000))

    def test_same_paid_payload_cannot_be_two_runs(self):
        control = load("control_timeline.json")
        treatment = load("treatment_timeline.json")
        treatment["paid_payload_sha256"] = control["paid_payload_sha256"]
        failures = comparator.check_match(control, treatment)
        self.assertIn("E_DUPLICATE_RUN", codes(failures))

    def test_paid_payload_digest_ignores_redundant_equal_samples(self):
        unsplit = [[0, 100], [20, 200]]
        split = [[0, 100], [10, 100], [20, 200]]
        self.assertEqual(
            comparator.paid_payload_digest(
                unsplit, ["P0", "P0"], 0, 20),
            comparator.paid_payload_digest(
                split, ["P0", "P0", "P0"], 0, 20))

    def test_paid_payload_digest_keeps_real_state_boundaries(self):
        samples = [[0, 100], [10, 100], [20, 200]]
        self.assertNotEqual(
            comparator.paid_payload_digest(
                samples, ["P0", "P0", "P0"], 0, 20),
            comparator.paid_payload_digest(
                samples, ["P0", "P8", "P8"], 0, 20))

    def test_raw_trace_cannot_be_spliced_between_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shutil.copytree(FIX, root, dirs_exist_ok=True)
            control = load("control_timeline.json")
            treatment = load("treatment_timeline.json")
            control["raw_artifact_id"] = treatment["raw_artifact_id"]
            control["raw_artifact_path"] = treatment["raw_artifact_path"]
            control["raw_artifact_sha256"] = treatment["raw_artifact_sha256"]
            reseal(control)
            self.assertIn(
                "E_RAW_BINDING",
                codes(comparator.validate_timeline(control, root)))

    def test_physical_label_is_not_expressible_in_comparison_v1(self):
        control = load("control_timeline.json")
        treatment = load("treatment_timeline.json")
        repetition = load("repetition_set.json")
        record = comparator.build_comparison(
            control, treatment, FIX, repetition)
        self.assertEqual(comparator.validate_comparison(
            record, control, treatment, repetition, FIX), [])
        record["result_label"] = comparator.LABEL_GPU
        reseal(record)
        self.assertIn("E_SCHEMA", codes(comparator.validate_comparison(
            record, control, treatment, repetition, FIX)))

    def test_comparison_arithmetic_is_recomputed_from_sources(self):
        control = load("control_timeline.json")
        treatment = load("treatment_timeline.json")
        repetition = load("repetition_set.json")
        for field, value in (("control_energy_nj", 1),
                             ("meets_ten_percent_gate", False),
                             ("control_lower_nj", 2)):
            record = comparator.build_comparison(
                control, treatment, FIX, repetition)
            record[field] = value
            reseal(record)
            failures = comparator.validate_comparison(
                record, control, treatment, repetition, FIX)
            self.assertIn("E_INCOHERENT", codes(failures), field)

    def test_comparison_validator_returns_source_failure_without_traceback(self):
        control = load("control_timeline.json")
        treatment = load("treatment_timeline.json")
        repetition = load("repetition_set.json")
        record = comparator.build_comparison(
            control, treatment, FIX, repetition)
        control["uncertainty_nj"] = control["energy_nj"]
        reseal(control)
        failures = comparator.validate_comparison(
            record, control, treatment, repetition, FIX)
        self.assertIn("E_UNCERTAINTY", codes(failures))

    def test_comparison_reason_is_derived_and_closed(self):
        control = load("control_timeline.json")
        treatment = load("treatment_timeline.json")
        repetition = load("repetition_set.json")
        record = comparator.build_comparison(
            control, treatment, FIX, repetition)
        record["reason_code"] = comparator.LABEL_SERVER
        reseal(record)
        self.assertIn("E_SCHEMA", codes(comparator.validate_comparison(
            record, control, treatment, repetition, FIX)))

        record = comparator.build_comparison(
            control, treatment, FIX, repetition)
        record["reason_code"] = "PAIR_ONLY_NO_AGGREGATE_CLAIM"
        reseal(record)
        self.assertIn("E_INCOHERENT", codes(comparator.validate_comparison(
            record, control, treatment, repetition, FIX)))

    def test_nvml_parser_preserves_integer_microseconds_and_milliwatts(self):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".csv", delete=False, encoding="ascii") as handle:
            handle.write("2026/07/15 00:00:00.000001, 25.88, P8, 210, 0\n")
            handle.write("2026/07/15 00:00:01.000002, 25.81, P8, 210, 0\n")
            path = handle.name
        try:
            from import_nvml_trace import parse_csv
            samples, pstates, utils = parse_csv(path)
        finally:
            os.unlink(path)
        self.assertEqual(samples, [[0, 25_880], [1_000_001, 25_810]])
        self.assertEqual(pstates, ["P8", "P8"])
        self.assertEqual(utils, [0, 0])

    def test_nvml_parser_rejects_excess_power_precision(self):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".csv", delete=False, encoding="ascii") as handle:
            handle.write("2026/07/15 00:00:00.000001, 25.1234, P8, 210, 0\n")
            handle.write("2026/07/15 00:00:01.000002, 25.88, P8, 210, 0\n")
            path = handle.name
        try:
            from import_nvml_trace import parse_csv
            with self.assertRaises(integrator.TimelineError) as ctx:
                parse_csv(path)
        finally:
            os.unlink(path)
        self.assertEqual(ctx.exception.code, "E_SCHEMA")

    def test_nvml_parser_rejects_hidden_trailing_columns(self):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".csv", delete=False, encoding="ascii") as handle:
            handle.write(
                "2026/07/15 00:00:00.000001, 25.88, P8, 210, 0, hidden\n")
            handle.write("2026/07/15 00:00:01.000002, 25.81, P8, 210, 0\n")
            path = handle.name
        try:
            from import_nvml_trace import parse_csv
            with self.assertRaises(integrator.TimelineError) as ctx:
                parse_csv(path)
        finally:
            os.unlink(path)
        self.assertEqual(ctx.exception.code, "E_SCHEMA")

    def test_builder_derives_repetition_binding(self):
        repetition = load("repetition_set.json")
        repetition["pairs"][0]["control_record_sha256"] = "a" * 64
        reseal(repetition)
        with self.assertRaises(comparator.ComparisonError) as ctx:
            comparator.build_comparison(
                load("control_timeline.json"), load("treatment_timeline.json"),
                FIX, repetition)
        self.assertEqual(ctx.exception.code, "E_BINDING")

    def test_builder_binds_repetition_scope_and_instrument(self):
        repetition = load("repetition_set.json")
        repetition["scope"] = "SERVER_WALL"
        reseal(repetition)
        with self.assertRaises(comparator.ComparisonError) as ctx:
            comparator.build_comparison(
                load("control_timeline.json"), load("treatment_timeline.json"),
                FIX, repetition)
        self.assertEqual(ctx.exception.code, "E_BINDING")

    def test_builder_rejects_malformed_timeline_without_traceback(self):
        with self.assertRaises(comparator.ComparisonError) as ctx:
            comparator.build_comparison(
                {}, load("treatment_timeline.json"), FIX,
                load("repetition_set.json"))
        self.assertEqual(ctx.exception.code, "TIMELINE_INVALID")

    def test_zero_wall_uncertainty_floor_is_rejected(self):
        capability = load("wall_capability.json")
        capability["uncertainty_floor_mw"] = 0
        reseal(capability)
        failures = comparator.check_wall_capability(
            capability, load("control_timeline.json"), FIX)
        self.assertIn("E_SCHEMA", codes(failures))


class ArchitectureSeparation(unittest.TestCase):
    """CONTRACT.md section 0, asserted rather than promised."""

    def test_aggregate_timeline_is_refused_by_the_additive_guard(self):
        control = load("control_timeline.json")
        with self.assertRaises(comparator.ComparisonError) as ctx:
            comparator.assert_not_additive_input(control)
        self.assertEqual(ctx.exception.code, "E_ADDITIVE_REUSE")
        self.assertIn("must never enter the additive", str(ctx.exception))

    def test_e2_imports_no_e1_decision_logic(self):
        # e2_canon reuses E1's canon by design. Importing E1's binder, boundary,
        # or validator would mean E2 inherited additive decision logic.
        forbidden = {"binder", "boundary", "validator", "exact", "checker",
                     "reference"}
        loaded = forbidden & set(sys.modules)
        self.assertEqual(loaded, set(),
                         f"E2 must not import E1 decision logic, found {loaded}")

    def test_e2_canon_agrees_byte_for_byte_with_e1(self):
        sys.path.insert(0, str(ROOT.parent / "s10_power_frontier_repair" / "evidence"))
        import canon as e1_canon
        for value in ({"a": 1}, {"z": [1, {"b": "x"}]}, {}, load("control_timeline.json")):
            self.assertEqual(canon.canonical(value), e1_canon.canonical(value))
            self.assertEqual(canon.digest(value), e1_canon.digest(value))


class ExistingTraceIsRejected(unittest.TestCase):
    """CP5: the real A6000 NVML trace, as a negative control."""

    TRACE = (ROOT.parent / "s10_power_frontier"
             / "artifacts" / "cp2_a6000_power_trace.csv")

    def test_the_real_trace_fails_the_frozen_update_gate(self):
        if not self.TRACE.exists():
            self.skipTest("historical A6000 trace not present")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "src" / "import_nvml_trace.py"),
             "--csv", str(self.TRACE), "--json"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["rows"], 323)
        # 57 independent sensor updates behind 323 rows: a 10 Hz poll of a 1 Hz
        # sensor. This is the number the gate turns on.
        self.assertEqual(report["independent_updates"], 57)
        self.assertLess(report["independent_updates"],
                        integrator.MIN_INDEPENDENT_UPDATES)
        self.assertEqual(report["scope"], "GPU_BOARD")
        joined = " ".join(report["failures"])
        self.assertIn("E_UPDATES", joined)
        self.assertIn("E_STATUS_CHANGE", joined)
        self.assertIn("E_PAIRS", joined)

    def test_the_gate_is_not_lowered_to_admit_it(self):
        self.assertEqual(integrator.MIN_INDEPENDENT_UPDATES, 100)
        self.assertGreater(integrator.MIN_INDEPENDENT_UPDATES, 57)


class SchemaGate(unittest.TestCase):
    def test_hostile_sitecustomize_cannot_replace_canonical_type_gate(self):
        trusted = (ROOT.parent / "s10_power_frontier_repair"
                   / "evidence" / "canon.py").resolve()
        with tempfile.TemporaryDirectory() as tmp:
            sitecustomize = pathlib.Path(tmp) / "sitecustomize.py"
            sitecustomize.write_text(
                "import importlib.util, sys, types\n"
                f"p = {str(trusted)!r}\n"
                "s = importlib.util.spec_from_file_location('_real_canon', p)\n"
                "r = importlib.util.module_from_spec(s)\n"
                "s.loader.exec_module(r)\n"
                "m = types.ModuleType('canon')\n"
                "for n in dir(r): setattr(m, n, getattr(r, n))\n"
                "m.is_int = lambda value: True\n"
                "m.check_int = lambda *args, **kwargs: None\n"
                "m.check_integers = lambda *args, **kwargs: None\n"
                "sys.modules['canon'] = m\n",
                encoding="ascii")
            program = (
                "import pathlib, sys\n"
                f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
                "import e2_canon\n"
                f"assert pathlib.Path(e2_canon._e1_canon.__file__).resolve() == "
                f"pathlib.Path({str(trusted)!r})\n"
                "try:\n"
                "    e2_canon.check_integers({'energy_nj': 1.0}, 'hostile')\n"
                "except ValueError:\n"
                "    pass\n"
                "else:\n"
                "    raise SystemExit(9)\n")
            proc = subprocess.run(
                ["/usr/bin/python3", "-c", program], capture_output=True,
                text=True,
                env={"PATH": "/usr/bin:/bin", "PYTHONPATH": tmp,
                     "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_hostile_pythonpath_cannot_replace_the_schema_engine(self):
        # The worker runs under `-I`, which ignores PYTHONPATH. Plant a hostile
        # jsonschema that would approve everything and assert it is not used.
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "jsonschema.py").write_text(
                "class Draft202012Validator:\n"
                "    def __init__(self, *a, **k): pass\n"
                "    @staticmethod\n"
                "    def check_schema(*a, **k): pass\n"
                "    def iter_errors(self, *a, **k): return []\n",
                encoding="ascii")
            env = {**os.environ, "PYTHONPATH": tmp}
            bad = {"schema_version": 999}
            proc = subprocess.run(
                ["/usr/bin/python3", "-I", str(ROOT / "src" / "e2_schema_gate.py")],
                input=json.dumps({"kind": "timeline", "document": bad}),
                capture_output=True, text=True, env=env)
            payload = json.loads(proc.stdout)
            self.assertNotIn("engine_error", payload)
            self.assertTrue(payload["errors"],
                            "a hostile PYTHONPATH jsonschema was used: the -I "
                            "isolation flag is not working")

    def test_unknown_document_kind_fails_closed(self):
        with self.assertRaises(comparator.ComparisonError) as ctx:
            comparator.schema_errors("not_a_kind", {})
        self.assertEqual(ctx.exception.code, "E_SCHEMA")

    def test_unsupported_version_fails_closed(self):
        control = load("control_timeline.json")
        control["schema_version"] = 2
        self.assertIn("E_SCHEMA",
                      codes(comparator.validate_timeline(control, FIX)))

    def test_unknown_field_fails_closed(self):
        control = load("control_timeline.json")
        control["surprise"] = 1
        self.assertIn("E_SCHEMA",
                      codes(comparator.validate_timeline(control, FIX)))

    def test_duplicate_json_keys_are_rejected(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                             encoding="ascii")
        handle.write('{"a": 1, "a": 2}')
        handle.close()
        with self.assertRaises(canon.DuplicateKeyError):
            canon.load_strict(handle.name)

    def test_nan_and_infinity_are_rejected(self):
        for literal in ('{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
            handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                                 encoding="ascii")
            handle.write(literal)
            handle.close()
            with self.assertRaises(ValueError):
                canon.load_strict(handle.name)

    def test_schemas_use_only_local_refs(self):
        for path in (ROOT / "schemas").glob("*.json"):
            text = path.read_text(encoding="ascii")
            document = json.loads(text)
            self._assert_local(document, path.name)

    def _assert_local(self, value, name):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "$ref":
                    self.assertTrue(item.startswith("#/"), f"{name}: {item}")
                self._assert_local(item, name)
        elif isinstance(value, list):
            for item in value:
                self._assert_local(item, name)


if __name__ == "__main__":
    unittest.main()
