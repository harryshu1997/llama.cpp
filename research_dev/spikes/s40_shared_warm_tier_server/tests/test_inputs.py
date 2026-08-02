#!/usr/bin/env python3

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
S40 = HERE.parent
sys.path.insert(0, str(S40))

from evidence_common import EvidenceError, canonical_bytes  # noqa: E402
import validate_inputs  # noqa: E402


class InputContractTests(unittest.TestCase):
    def load_contract(self):
        return json.loads(
            (S40 / "EXPERIMENT_CONTRACT.json").read_text(encoding="ascii"))

    def validate_mutation(self, value):
        with tempfile.TemporaryDirectory(prefix="s40_contract_") as directory:
            path = Path(directory) / "contract.json"
            for key in ("workload", "baselines"):
                for name, item in value[key].items():
                    if name.endswith("_path"):
                        item_path = (S40 / item).resolve()
                        item = str(item_path)
                        value[key][name] = item
            c0 = value["baselines"]["C0_EXISTING_GPU_SWITCH"]
            for key in list(c0):
                if key.endswith("_path"):
                    c0[key] = str((S40 / c0[key]).resolve())
            runtime_source = value["runtime_source"]
            runtime_source["desktop_contract_path"] = str(
                (S40 / runtime_source["desktop_contract_path"]).resolve())
            path.write_bytes(canonical_bytes(value))
            return validate_inputs.validate_contract(path)

    def test_frozen_inputs_and_c0_validate(self):
        result = validate_inputs.validate_contract()
        self.assertEqual(result["status"], "S40_INPUTS_VALID")
        self.assertEqual(result["request_count"], 74)
        self.assertEqual(result["switch_count"], 9)
        self.assertEqual(sorted(result["models"].values()), [17, 57])

    def test_contract_is_canonical(self):
        path = S40 / "EXPERIMENT_CONTRACT.json"
        value = self.load_contract()
        self.assertEqual(path.read_bytes(), canonical_bytes(value))

    def test_greedy_sampling_is_load_bearing(self):
        value = self.load_contract()
        value["workload"]["greedy_sampling"] = False
        with self.assertRaisesRegex(EvidenceError, "greedy sampling"):
            self.validate_mutation(value)

    def test_model_order_is_load_bearing(self):
        value = self.load_contract()
        value["workload"]["models"].reverse()
        with self.assertRaisesRegex(EvidenceError, "model order"):
            self.validate_mutation(value)

    def test_historical_switches_do_not_drive_new_modes(self):
        value = self.load_contract()
        value["policy"]["historical_switch_rows_drive_new_modes"] = True
        with self.assertRaisesRegex(EvidenceError, "contract.policy"):
            self.validate_mutation(value)

    def test_c3_is_one_gpu_partial_offload(self):
        value = self.load_contract()
        value["matrix"]["C3_DUAL_PARTIAL_OFFLOAD"][
            "executor_placement"
        ] = "TWO_GPU_PARTIAL_PLUS_CPU"
        with self.assertRaisesRegex(EvidenceError, "matrix.C3"):
            self.validate_mutation(value)

    def test_t2_no_promotion_is_load_bearing(self):
        value = self.load_contract()
        value["matrix"]["T2_PHONE_NO_PROMOTION"][
            "executor_placement"
        ] = "GPU_HOT_OP15_OP12_ALTERNATE"
        with self.assertRaisesRegex(EvidenceError, "matrix.T2"):
            self.validate_mutation(value)

    def test_c1_has_no_warm_executor(self):
        value = self.load_contract()
        value["matrix"]["C1_GPU_ONLY_OPTIMIZED"][
            "warm_executor"
        ] = "FAKE"
        with self.assertRaisesRegex(EvidenceError, "matrix.C1"):
            self.validate_mutation(value)

    def test_energy_scope_is_load_bearing(self):
        value = self.load_contract()
        value["energy"]["allowed_scope"] = "SERVER_WALL"
        with self.assertRaisesRegex(EvidenceError, "energy"):
            self.validate_mutation(value)

    def test_c0_policy_label_is_load_bearing(self):
        value = self.load_contract()
        value["baselines"]["C0_EXISTING_GPU_SWITCH"]["policy"] = "coalescing"
        with self.assertRaisesRegex(EvidenceError, "C0 policy"):
            self.validate_mutation(value)

    def test_unknown_mode_is_rejected(self):
        value = self.load_contract()
        value["matrix"]["UNKNOWN"] = copy.deepcopy(
            value["matrix"]["C1_GPU_ONLY_OPTIMIZED"])
        with self.assertRaisesRegex(EvidenceError, "mode set"):
            self.validate_mutation(value)

    def test_contract_digest_is_required_in_future_runs(self):
        value = self.load_contract()
        value["run_manifest_requirements"][
            "experiment_contract_sha256_required"
        ] = False
        with self.assertRaisesRegex(EvidenceError, "run_manifest_requirements"):
            self.validate_mutation(value)

    def test_http_worker_capacity_is_frozen(self):
        value = self.load_contract()
        value["run_manifest_requirements"]["required_http_threads"] = 8
        with self.assertRaisesRegex(
                EvidenceError, "run_manifest_requirements"):
            self.validate_mutation(value)


if __name__ == "__main__":
    unittest.main()
