#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_residency_v1.analyze_gpu_tensor_manifests import (  # noqa: E402
    DEFAULT_ABBA,
    DEFAULT_CAPACITY,
    DEFAULT_GEMMA_MANIFEST,
    DEFAULT_QWEN_MANIFEST,
    TensorManifestBundleError,
    analyze,
    build,
    canonical,
    read_object,
    sha256,
)


RESULT = (
    ROOT
    / "dynamic_residency_v1/results/GPU_TENSOR_MANIFEST_BUNDLE_V1.json"
)


def inputs() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, str],
]:
    paths = {
        "capacity_file_sha256": DEFAULT_CAPACITY,
        "gemma_manifest_file_sha256": DEFAULT_GEMMA_MANIFEST,
        "qwen_manifest_file_sha256": DEFAULT_QWEN_MANIFEST,
        "service_abba_file_sha256": DEFAULT_ABBA,
    }
    return (
        read_object(DEFAULT_QWEN_MANIFEST),
        read_object(DEFAULT_GEMMA_MANIFEST),
        read_object(DEFAULT_CAPACITY),
        read_object(DEFAULT_ABBA),
        {name: sha256(path) for name, path in paths.items()},
    )


class GpuTensorManifestBundleTests(unittest.TestCase):
    def test_checked_bundle_matches_raw_receipts(self) -> None:
        self.assertEqual(read_object(RESULT), build())

    def test_exact_transition_slice_is_bound(self) -> None:
        result = build()
        transition = result["transition_slice"]
        self.assertEqual(transition["qwen_evicted_layer_ids"], [23, 24, 25])
        self.assertEqual(transition["qwen_evicted_tensor_count"], 33)
        self.assertEqual(
            transition["qwen_evicted_raw_tensor_bytes"], 1_981_934_592
        )
        self.assertEqual(
            transition["gemma_added_raw_tensor_bytes"], 2_013_281_280
        )
        self.assertEqual(transition["net_added_raw_tensor_bytes"], 31_346_688)

    def test_runtime_model_buffers_match_raw_tensor_bytes(self) -> None:
        result = build()
        observations = result["runtime_allocation_binding"]
        self.assertEqual(
            observations["qwen_gpu_18"]["logged_model_buffer_mib"],
            [12194.45, 12194.45],
        )
        self.assertEqual(
            observations["gemma_gpu_1"]["logged_model_buffer_mib"],
            [1920.01, 1920.01, 1920.01],
        )
        self.assertTrue(
            result["claim_gates"][
                "runtime_model_buffer_bound_to_raw_tensor_bytes"
            ]
        )

    def test_final_fit_does_not_imply_atomic_staging_fit(self) -> None:
        result = build()
        staging = result["atomic_staging"]
        self.assertTrue(staging["final_measured_placement_fits"])
        self.assertFalse(staging["atomic_stage_before_evict_fits"])
        self.assertEqual(
            staging["stageable_bytes_beyond_reserve"], 205_520_896
        )
        self.assertEqual(staging["staging_shortfall_bytes"], 2_238_709_760)
        self.assertTrue(staging["gemma_after_qwen15_atomic_stage_fits"])
        self.assertEqual(
            staging["qwen15_stageable_bytes_beyond_reserve"],
            2_624_585_728,
        )
        self.assertEqual(
            staging["gemma_after_qwen15_staging_margin_bytes"],
            180_355_072,
        )
        self.assertEqual(
            [
                row["mode"]
                for row in staging["required_transition_sequence"]
            ],
            [
                "DRAIN_OR_EVICT_BEFORE_STAGE_WITH_READY_FALLBACK",
                "ATOMIC_STAGE_BEFORE_EVICT",
            ],
        )
        self.assertEqual(
            staging["required_transition_mode"],
            "DRAIN_OR_EVICT_BEFORE_STAGE_WITH_READY_FALLBACK",
        )

    def test_record_hash_is_canonical(self) -> None:
        result = build()
        claimed = result.pop("record_sha256")
        self.assertEqual(claimed, hashlib.sha256(canonical(result)).hexdigest())

    def test_rehashed_model_identity_tamper_fails_closed(self) -> None:
        qwen, gemma, capacity, abba, evidence = inputs()
        changed = deepcopy(qwen)
        changed["model"]["id"] = "different-model"
        changed.pop("record_sha256")
        changed["record_sha256"] = hashlib.sha256(
            canonical(changed)
        ).hexdigest()
        with self.assertRaisesRegex(
            TensorManifestBundleError, "source and model identity"
        ):
            analyze(
                changed,
                gemma,
                capacity,
                abba,
                evidence=evidence,
            )

    def test_tensor_table_tamper_fails_placement_binding(self) -> None:
        qwen, gemma, capacity, abba, evidence = inputs()
        changed = deepcopy(gemma)
        changed["selected_tensors"][0]["raw_sha256"] = "0" * 64
        changed.pop("record_sha256")
        changed["record_sha256"] = hashlib.sha256(
            canonical(changed)
        ).hexdigest()
        with self.assertRaisesRegex(
            TensorManifestBundleError, "placement tensor binding"
        ):
            analyze(
                qwen,
                changed,
                capacity,
                abba,
                evidence=evidence,
            )


if __name__ == "__main__":
    unittest.main()
