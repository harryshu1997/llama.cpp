#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_residency_v1.analyze_adoption_qualification import (  # noqa: E402
    DEFAULT_RESULT_ROOT,
    SCHEMA,
    build,
)
from dynamic_residency_v1.analyze_prefetch_fence_run import (  # noqa: E402
    AnalysisError,
    canonical,
)
from dynamic_residency_v1.analyze_prefetch_qualification import (  # noqa: E402
    verify_canonical_record,
)


RESULT = DEFAULT_RESULT_ROOT / "ADOPTION_QUALIFICATION_V1.json"


class AdoptionQualificationTests(unittest.TestCase):
    def test_checked_result_matches_raw_receipts(self) -> None:
        self.assertEqual(
            json.loads(RESULT.read_text(encoding="ascii")),
            build(),
        )

    def test_same_process_adoption_is_qualified(self) -> None:
        result = build()
        self.assertEqual(
            result["admission"],
            "SAME_PROCESS_WEIGHT_ADOPTION_QUALIFIED_ENERGY_ABBA_PENDING",
        )
        self.assertTrue(
            result["claim"]["same_process_weight_adoption_qualified"]
        )
        self.assertIsNone(
            result["claim"]["incremental_dynamic_energy_savings_pct"]
        )

    def test_exact_output_reserve_and_no_swap(self) -> None:
        result = build()
        self.assertEqual(
            result["gemma_execution"]["tokens_sha256"],
            "b902347e9dae1ce33fe87cc507b5610b"
            "bbc0de62d93629cee9a8d7f45a73eec6",
        )
        self.assertEqual(result["memory"]["swap_limit_bytes"], 0)
        self.assertGreaterEqual(
            result["weight_adoption"]["gpu_reserve_margin_bytes"], 0
        )
        self.assertTrue(result["gates"]["process_swap_zero"])

    def test_gpu_is_not_continuously_busy(self) -> None:
        result = build()
        gpu = result["gpu_utilization_during_qwen"]
        self.assertFalse(result["claim"]["gpu_continuously_busy"])
        self.assertEqual(gpu["samples_at_or_above_90_pct"], 0)
        self.assertLess(gpu["mean_pct"], 90)

    def test_zero_swap_semantic_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "result"
            shutil.copytree(DEFAULT_RESULT_ROOT, copied)
            path = (
                copied
                / "raw/adoptable-qualified-r5.adoption/"
                "PROCESS_MEMORY_V1.json"
            )
            value = json.loads(path.read_text(encoding="ascii"))
            value["processes"]["qwen"]["swap_bytes"] = 1
            value.pop("record_sha256")
            value["record_sha256"] = hashlib.sha256(
                canonical(value)
            ).hexdigest()
            path.write_bytes(canonical(value))
            with self.assertRaisesRegex(AnalysisError, "zero process swap"):
                build(copied)

    def test_record_hash_tamper_fails_closed(self) -> None:
        value = build()
        changed = deepcopy(value)
        changed["claim"]["gpu_continuously_busy"] = True
        with self.assertRaisesRegex(AnalysisError, "canonical hash"):
            verify_canonical_record(changed, SCHEMA)


if __name__ == "__main__":
    unittest.main()
