#!/usr/bin/env python3

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checker"))

import checker  # noqa: E402


def contribution(index):
    return {
        "pair_index": index,
        "control_energy_nj": 1000,
        "control_uncertainty_nj": 1,
        "treatment_energy_nj": 800,
        "treatment_uncertainty_nj": 1,
    }


def aggregate_record(scope="GPU_BOARD", anchor_kind="RFC3161_TSA"):
    contributions = [contribution(index) for index in range(8)]
    c, u_c, t, u_t = checker.sum_pairs(contributions)
    control_lower, treatment_upper, relief, meets = checker.decide(
        c, u_c, t, u_t)
    anchor_property = (
        "ORDERING_AND_ENUMERABLE"
        if anchor_kind == "TRANSPARENCY_LOG_INCLUSION"
        else "ORDERING_ONLY"
    )
    if anchor_property == "ORDERING_AND_ENUMERABLE":
        result_label = checker.SCOPE_LABEL[scope]
        reason_code = "ALL_PAIRS_CONSERVATIVE_RELIEF"
    else:
        result_label = checker.LABEL_INVALID
        reason_code = "ANCHOR_NOT_ENUMERABLE"
    server_delta = c - t if scope == "SERVER_WALL" else None
    record = {
        "schema_version": 1,
        "kind": "AggregateComparison",
        "aggregate_id": "agg.test",
        "chain_id": "chain",
        "set_id": "set",
        "plan_id": "plan",
        "plan_sha256": "a" * 64,
        "plan_anchor_record_sha256": "b" * 64,
        "ledger_close_record_sha256": "c" * 64,
        "ledger_head_sha256": "d" * 64,
        "request_set_manifest_sha256": "e" * 64,
        "aggregate_method": "SUM_ALL_PAIRS_V1",
        "anchor_kind": anchor_kind,
        "anchor_property": anchor_property,
        "scope": scope,
        "instrument_kind": "SYNTHETIC",
        "n_pairs": len(contributions),
        "pair_contributions": contributions,
        "control_energy_sum_nj": c,
        "control_uncertainty_sum_nj": u_c,
        "treatment_energy_sum_nj": t,
        "treatment_uncertainty_sum_nj": u_t,
        "control_lower_nj": control_lower,
        "treatment_upper_nj": treatment_upper,
        "conservative_margin_nj": control_lower - treatment_upper,
        "boundary_delta_nj": c - t,
        "server_wall_delta_nj": server_delta,
        "phone_plus_external_break_even_budget_nj": server_delta,
        "conservative_relief": relief and meets,
        "meets_ten_percent_gate": meets,
        "result_label": result_label,
        "reason_code": reason_code,
        "record_sha256": "",
    }
    record["record_sha256"] = checker.record_digest(record)
    return record


def reseal(record):
    record["record_sha256"] = checker.record_digest(record)


class CheckerHardening(unittest.TestCase):
    def test_physical_claim_branch_is_refused(self):
        record = aggregate_record(
            anchor_kind="TRANSPARENCY_LOG_INCLUSION")
        with self.assertRaises(checker.CheckError) as caught:
            checker.check_aggregate(record)
        self.assertEqual(caught.exception.code, "E_LABEL")
        message = str(caught.exception)
        self.assertNotIn(checker.LABEL_GPU, message)
        self.assertNotIn(checker.LABEL_SERVER, message)

    def test_every_derived_integer_rejects_float_and_bool(self):
        scalar_fields = (
            "n_pairs",
            "control_energy_sum_nj",
            "control_uncertainty_sum_nj",
            "treatment_energy_sum_nj",
            "treatment_uncertainty_sum_nj",
            "control_lower_nj",
            "treatment_upper_nj",
            "conservative_margin_nj",
            "boundary_delta_nj",
            "server_wall_delta_nj",
            "phone_plus_external_break_even_budget_nj",
        )
        for field in scalar_fields:
            for bad_value in (1.0, True):
                with self.subTest(field=field, value=bad_value):
                    record = aggregate_record(scope="SERVER_WALL")
                    record[field] = bad_value
                    reseal(record)
                    with self.assertRaises(checker.CheckError) as caught:
                        checker.check_aggregate(record)
                    self.assertEqual(caught.exception.code, "E_TYPE")

        for bad_value in (0.0, False):
            with self.subTest(field="pair_index", value=bad_value):
                record = aggregate_record()
                record["pair_contributions"][0]["pair_index"] = bad_value
                reseal(record)
                with self.assertRaises(checker.CheckError) as caught:
                    checker.check_aggregate(record)
                self.assertEqual(caught.exception.code, "E_TYPE")

        pair_fields = (
            "control_energy_nj",
            "control_uncertainty_nj",
            "treatment_energy_nj",
            "treatment_uncertainty_nj",
        )
        for field in pair_fields:
            for bad_value, expected in (
                    (1.0, "E_TYPE"), (True, "E_TYPE"),
                    (checker.MAX_INT + 1, "E_OVERFLOW")):
                with self.subTest(field=field, value=bad_value):
                    record = aggregate_record()
                    record["pair_contributions"][0][field] = bad_value
                    reseal(record)
                    with self.assertRaises(checker.CheckError) as caught:
                        checker.check_aggregate(record)
                    self.assertEqual(caught.exception.code, expected)

    def test_cli_never_echoes_a_physical_label(self):
        records = [
            aggregate_record(anchor_kind="TRANSPARENCY_LOG_INCLUSION"),
            aggregate_record(),
        ]
        records[1]["kind"] = checker.LABEL_SERVER
        reseal(records[1])
        for index, record in enumerate(records):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "aggregate.json"
                path.write_text(json.dumps(record), encoding="ascii")
                result = subprocess.run(
                    [sys.executable, str(ROOT / "checker" / "checker.py"),
                     "--aggregate", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            combined = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(checker.LABEL_GPU, combined)
            self.assertNotIn(checker.LABEL_SERVER, combined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
