#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
for path in (REPO_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_residency_v1.analyze_dual_residency_pair import (  # noqa: E402
    AnalysisError,
    DEFAULT_CONTROL,
    DEFAULT_TREATMENT,
    analyze,
    canonical,
    read_object,
    validate_arm,
)


RESULT = (
    ROOT
    / "dynamic_residency_v1/results/DUAL_RESIDENCY_PAIR_DIAGNOSTIC_V1.json"
)


class DynamicResidencyPairTests(unittest.TestCase):
    def test_checked_in_result_matches_analyzer(self) -> None:
        self.assertEqual(
            read_object(RESULT), analyze(DEFAULT_CONTROL, DEFAULT_TREATMENT)
        )

    def test_single_pair_is_promising_but_not_admitted(self) -> None:
        result = analyze(DEFAULT_CONTROL, DEFAULT_TREATMENT)
        self.assertEqual(result["status"], "PROMISING_SINGLE_PAIR_NO_CLAIM")
        self.assertEqual(
            result["observed_direction"],
            "LOWER_SERVER_ENERGY_AND_WALL_SINGLE_PAIR",
        )
        self.assertAlmostEqual(
            result["changes"]["server_energy_pct"], -9.651141891305992
        )
        self.assertAlmostEqual(
            result["changes"]["wall_service_pct"], -5.511601553245349
        )
        self.assertFalse(result["admission"]["eligible"])
        self.assertIsNone(result["admission"]["dynamic_energy_claim"])
        self.assertEqual(
            result["admission"]["decision"],
            "REPEAT_ABBA_BEFORE_ADMISSION",
        )
        self.assertTrue(all(result["validity_gates"].values()))
        self.assertFalse(any(result["claim_gates"].values()))

    def test_energy_moves_from_cpu_to_gpu(self) -> None:
        result = analyze(DEFAULT_CONTROL, DEFAULT_TREATMENT)
        changes = result["changes"]
        self.assertAlmostEqual(
            changes["cpu_package_energy_pct"], -25.276716492461983
        )
        self.assertAlmostEqual(
            changes["gpu_board_energy_pct"], 44.85626957135806
        )
        self.assertLess(changes["server_energy_pct"], 0)
        self.assertEqual(result["arms"]["control"]["energy"], {
            "cpu_package_j": 5498.226403,
            "gpu_board_j": 1576.170002,
            "server_j": 7074.396405,
        })
        self.assertEqual(result["arms"]["treatment"]["energy"], {
            "cpu_package_j": 4108.455303,
            "gpu_board_j": 2283.181067,
            "server_j": 6391.63637,
        })

    def test_quality_is_reported_without_claiming_exact_gemma(self) -> None:
        quality = analyze(DEFAULT_CONTROL, DEFAULT_TREATMENT)["quality"]
        self.assertEqual(quality["qwen"], {
            "common_prefix_tokens": 9,
            "exact": True,
            "positional_agreement_pct": 100.0,
            "positional_matches": 9,
            "tokens": 9,
        })
        self.assertEqual(quality["gemma"]["common_prefix_tokens"], 36)
        self.assertEqual(quality["gemma"]["positional_matches"], 37)
        self.assertAlmostEqual(
            quality["gemma"]["positional_agreement_pct"],
            90.2439024390244,
        )
        self.assertFalse(quality["gemma"]["exact"])

    def test_record_hash_is_canonical(self) -> None:
        result = analyze(DEFAULT_CONTROL, DEFAULT_TREATMENT)
        claimed = result.pop("record_sha256")
        self.assertEqual(claimed, hashlib.sha256(canonical(result)).hexdigest())

    def test_manifest_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "control"
            shutil.copytree(DEFAULT_CONTROL, root)
            with (root / "qwen-smoke.raw").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(AnalysisError, "manifest digest"):
                validate_arm(root, "control")

    def test_input_epoch_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "treatment"
            shutil.copytree(DEFAULT_TREATMENT, root)
            result = deepcopy(read_object(root / "RESULT.json"))
            result["inputs"]["trace"]["sha256"] = "0" * 64
            unsigned = dict(result)
            unsigned.pop("record_sha256")
            result["record_sha256"] = hashlib.sha256(
                canonical(unsigned)
            ).hexdigest()
            result_path = root / "RESULT.json"
            result_path.write_bytes(canonical(result))
            manifest_path = root / "SHA256SUMS.txt"
            lines = manifest_path.read_text(encoding="ascii").splitlines()
            result_digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
            rewritten = [
                f"{result_digest}  RESULT.json"
                if line.endswith("  RESULT.json") else line
                for line in lines
            ]
            manifest_path.write_text("\n".join(rewritten) + "\n",
                                     encoding="ascii")
            with self.assertRaisesRegex(
                AnalysisError, "input identity trace"
            ):
                validate_arm(root, "treatment")


if __name__ == "__main__":
    unittest.main()
