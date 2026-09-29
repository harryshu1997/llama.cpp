#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fit_model_energy  # noqa: E402


COEFFICIENTS = {
    "duration_ms": 10,
    "input_token_rows": 2,
    "output_token_rows": 3,
    "decode_steps": 4,
    "prefill_ubatches": 5,
}


def sample(index: int, holdout: bool = False) -> dict:
    features = {
        "duration_ms": 100 + 7 * index,
        "input_token_rows": 20 + index * index,
        "output_token_rows": 3 + 2 * index,
        "decode_steps": 1 + ((index * index + 3 * index) % 11),
        "prefill_ubatches": 2 + (index % 3),
    }
    actual = sum(COEFFICIENTS[name] * value for name, value in features.items())
    return {
        "case_id": f"case-{index}",
        "components_uj": {"server_total": actual},
        "features": features,
        "holdout": holdout,
        "kind": "inference",
        "repeat_index": 1,
    }


class ModelEnergyFitTest(unittest.TestCase):
    def test_assumed_phone_energy(self) -> None:
        self.assertEqual(
            fit_model_energy.assumed_phone_energy_uj(5.0, 2.0),
            10_000_000,
        )

    def test_assumed_phone_energy_is_labeled_estimated(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            result_path = root / "RESULT.json"
            result_path.write_text(json.dumps({
                "cases": [{
                    "case": {
                        "case_id": "decode",
                        "holdout": False,
                        "kind": "inference",
                    },
                    "duration_s": 2.0,
                    "features": {
                        "decode_steps": 1,
                        "input_token_rows": 32,
                        "output_token_rows": 1,
                        "prefill_ubatches": 1,
                    },
                    "server_energy": {
                        "boundary": "paid_trace_interval",
                        "cpu_package_energy_j": 20.0,
                        "gpu_board_energy_j": 10.0,
                        "server_compute_device_energy_j": 30.0,
                    },
                    "target_duration_met": True,
                }],
                "preflight": {
                    "case_plan_sha256": "cases",
                    "gpu": {"uuid": "gpu"},
                    "model_file_bytes": 1,
                    "model_id": "model",
                    "model_catalog_sha256": "catalog",
                    "model_sha256": "model-sha",
                    "placement": "CPU+OP15-HTP",
                    "route": "gemma-op15",
                    "runtime_manifest": [],
                    "server_sha256": "server-sha",
                },
                "repeat_index": 1,
                "schema": "s42-model-energy-result-v1",
                "status": "PASS",
            }), encoding="ascii")
            rows, metadata = fit_model_energy.observations(
                [result_path], assumed_phone_power_w=5.0
            )
        self.assertEqual(rows[0]["components_uj"]["phone"], 10_000_000)
        self.assertEqual(
            rows[0]["components_uj"]["accounted_fleet"],
            40_000_000,
        )
        self.assertEqual(metadata["component_status"]["server_total"], "measured")
        self.assertEqual(metadata["component_status"]["phone"], "estimated")
        self.assertEqual(metadata["phone_assumption_w"], 5.0)

    def test_exact_nonnegative_fit_and_holdout(self) -> None:
        rows = [sample(index) for index in range(1, 8)]
        rows.append(sample(8, holdout=True))
        value = fit_model_energy.fit_nonnegative(rows, "server_total")
        self.assertEqual(value["coefficients_uj_per_unit"], COEFFICIENTS)
        self.assertEqual(value["training"]["max_abs_error_pct"], 0.0)
        self.assertEqual(value["holdout"]["max_abs_error_pct"], 0.0)

    def test_fit_is_nonnegative(self) -> None:
        rows = [sample(index) for index in range(1, 8)]
        rows.append(sample(8, holdout=True))
        rows[1]["components_uj"]["server_total"] //= 2
        value = fit_model_energy.fit_nonnegative(rows, "server_total")
        self.assertTrue(all(
            coefficient >= 0
            for coefficient in value["coefficients_uj_per_unit"].values()
        ))

    def test_training_coverage_is_required(self) -> None:
        rows = [sample(index) for index in range(1, 4)]
        rows.append(sample(4, holdout=True))
        with self.assertRaises(fit_model_energy.FitError):
            fit_model_energy.fit_nonnegative(rows, "server_total")


if __name__ == "__main__":
    unittest.main()
