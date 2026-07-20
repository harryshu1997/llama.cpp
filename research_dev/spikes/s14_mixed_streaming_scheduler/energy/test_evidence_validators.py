#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest import mock

import validate_cp_b_result as cpb
import validate_cp_d_result as cpd
import validate_cp_f_result as cpf
from bge_corpus import make_prompt


HERE = Path(__file__).resolve().parent


class EvidenceValidatorTests(unittest.TestCase):
    def test_cp_f_rejects_changed_live_binary(self) -> None:
        result = cpf.load(HERE / "cp_f_live_op15_result.json")
        with self.assertRaisesRegex(cpf.ValidationError, "artifact digest mismatch llama_layersplit_host"):
            cpf.validate(result)

    def test_cp_f_inner_replay_rejects_summary_and_placement_mutations(self) -> None:
        result = cpf.load(HERE / "cp_f_live_op15_result.json")
        bad_summary = copy.deepcopy(result)
        bad_summary["summary"]["selected_gpu_energy_saving_frac"] = 0.9
        bad_placement = copy.deepcopy(result)
        treatment = next(row for row in bad_placement["rows"] if row["label"] == "P2")
        treatment["gemma"]["op15_cert"]["layer_end"] = 7
        profile_id = result["profile"]["op15"]
        with mock.patch.object(cpf, "check_artifacts", return_value=profile_id):
            with self.assertRaisesRegex(cpf.ValidationError, "summary mismatch"):
                cpf.validate(bad_summary)
            with self.assertRaisesRegex(cpf.VE.EvidenceError, "range mismatch"):
                cpf.validate(bad_placement)

    def test_shared_corpus_matches_the_measured_server_input(self) -> None:
        measured = (HERE / "logs_bge_server" / "corr_32.txt").read_text()
        self.assertEqual(measured, make_prompt(25) + "\n")

    def test_current_cp_b_is_rejected_as_unmatched_and_provisional(self) -> None:
        phone = cpb.load_json(HERE / "cp_b_phone_bge_result.json")
        server = cpb.load_json(HERE / "bge_server_result.json")
        digest = hashlib.sha256((HERE / "bge_server_result.json").read_bytes()).hexdigest()
        issues = cpb.validate(phone, server, digest)
        self.assertIn("E_PROCS", issues)
        self.assertTrue(any(issue.startswith("E_SHAPE_MATCH") for issue in issues))
        self.assertIn("E_THERMAL:op12", issues)
        self.assertTrue(any(issue.startswith("E_COV") for issue in issues))

    def test_npu_thermal_snapshot_contract(self) -> None:
        good = {
            "valid": True,
            "sensor_class": "nsphmx-*",
            "sensors_millic": {"nsphmx-0": 32_000, "nsphmx-1": 33_000},
            "max_millic": 33_000,
        }
        self.assertTrue(cpb.thermal_valid(good, 85_000))
        bad = copy.deepcopy(good)
        bad["max_millic"] = 0
        self.assertFalse(cpb.thermal_valid(bad, 85_000))

    def test_cp_d_recomputes_the_measured_result(self) -> None:
        result = cpd.load_json(HERE / "cp_d_result.json")
        self.assertAlmostEqual(cpd.validate(result), 0.12161753630601269)

    def test_cp_d_rejects_energy_mutation(self) -> None:
        result = cpd.load_json(HERE / "cp_d_result.json")
        bad = copy.deepcopy(result)
        bad["measured"]["plan_rows"][0]["selected_gpu_energy_j"] += 1.0
        with self.assertRaisesRegex(cpd.EvidenceError, "does not sum"):
            cpd.validate(bad)

    def test_raw_power_replay_rejects_mutation(self) -> None:
        leg = {
            "paid_window_us": {"start": 100, "end": 300},
            "raw_power_samples": [
                {"t_us": 0, "power_mw": 10},
                {"t_us": 100, "power_mw": 20},
                {"t_us": 200, "power_mw": 30},
                {"t_us": 300, "power_mw": 40},
            ],
            "energy_nj": 5000,
            "energy_j": 0.000005,
        }
        self.assertEqual(cpd.replay_energy(leg), 5000)
        leg["energy_nj"] = 4999
        with self.assertRaisesRegex(cpd.EvidenceError, "replay"):
            cpd.replay_energy(leg)

    def test_duplicate_json_key_is_rejected(self) -> None:
        path = HERE / "_duplicate_test.json"
        try:
            path.write_text('{"schema":"a","schema":"b"}')
            with self.assertRaisesRegex(cpd.EvidenceError, "duplicate key"):
                cpd.load_json(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
