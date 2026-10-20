"""The preflight's trace coverage (2026-10-08): a single-model trace leaves the cold model idle."""

import unittest

from research_dev.scheduler.campaigns.burstgpt import preflight

ROLE_MODELS = {"hot": "qwen3-32b-q4_0", "cold": "gemma-4-12b-it-q4_0"}


class IdleTraceModelTests(unittest.TestCase):
    def test_a_role_declared_with_zero_rows_is_idle(self) -> None:
        manifest = {"execution_models": {"hot": 13, "cold": 0, "llama-3.2-1b-instruct-q4_0": 1}}
        self.assertEqual(preflight._idle_trace_models(manifest, ROLE_MODELS), {"gemma-4-12b-it-q4_0"})

    def test_mixed_and_legacy_manifests_leave_no_model_idle(self) -> None:
        self.assertEqual(preflight._idle_trace_models({"execution_models": {"hot": 7, "cold": 6}}, ROLE_MODELS), set())
        self.assertEqual(preflight._idle_trace_models({"base_trace": {"record_count": 74}}, ROLE_MODELS), set())


if __name__ == "__main__":
    unittest.main()
