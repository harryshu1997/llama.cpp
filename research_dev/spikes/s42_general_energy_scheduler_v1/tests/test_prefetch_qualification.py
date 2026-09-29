#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_residency_v1.analyze_prefetch_fence_run import (  # noqa: E402
    AnalysisError,
    canonical,
)
from dynamic_residency_v1.analyze_prefetch_qualification import (  # noqa: E402
    DEFAULT_RESULT_ROOT,
    build,
    verify_canonical_record,
)


RESULT = DEFAULT_RESULT_ROOT / "PREFETCH_QUALIFICATION_V1.json"


class PrefetchQualificationTests(unittest.TestCase):
    def test_checked_result_matches_raw_receipts(self) -> None:
        self.assertEqual(
            json.loads(RESULT.read_text(encoding="ascii")),
            build(),
        )

    def test_transfer_is_qualified_but_not_adoptable(self) -> None:
        result = build()
        self.assertEqual(
            result["admission"],
            "FENCE_TRANSFER_QUALIFIED_NO_WEIGHT_ADOPTION",
        )
        self.assertTrue(result["claim"]["transfer_mechanism_qualified"])
        self.assertFalse(
            result["claim"]["weight_adoptable_by_gemma_executor"]
        )
        self.assertIsNone(
            result["claim"]["incremental_dynamic_energy_savings_pct"]
        )

    def test_full_tied_output_tensor_is_verified_inside_reserve(self) -> None:
        tensor = build()["full_tied_output_tensor"]
        self.assertEqual(tensor["copied_bytes"], 2_013_265_920)
        self.assertEqual(tensor["copy_windows"], 54)
        self.assertEqual(tensor["window_overrun_max_ms"], 0)
        self.assertGreaterEqual(tensor["gpu_reserve_margin_bytes"], 0)

    def test_gpu_is_not_continuously_busy(self) -> None:
        result = build()
        self.assertFalse(result["claim"]["gpu_continuously_busy"])
        self.assertTrue(all(
            row["samples_at_or_above_90_pct"] == 0
            for row in result["gpu_utilization"].values()
        ))

    def test_record_hash_tamper_fails_closed(self) -> None:
        value = build()
        changed = deepcopy(value)
        changed["claim"]["gpu_continuously_busy"] = True
        with self.assertRaisesRegex(AnalysisError, "canonical hash"):
            verify_canonical_record(changed, changed["schema"])
        changed.pop("record_sha256")
        changed["record_sha256"] = hashlib.sha256(
            canonical(changed)
        ).hexdigest()
        verify_canonical_record(changed, changed["schema"])


if __name__ == "__main__":
    unittest.main()
