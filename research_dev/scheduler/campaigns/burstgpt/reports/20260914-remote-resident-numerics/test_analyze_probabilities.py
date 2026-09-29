import json
from pathlib import Path
import tempfile
import unittest

from analyze_probabilities import compare, read_stream


def stream(path, tokens, *, probabilities=True):
    rows = []
    for token in tokens:
        row = {"tokens": [token], "content": str(token), "stop": False}
        if probabilities:
            row["completion_probabilities"] = [{
                "id": token,
                "top_logprobs": [{"id": key, "token": str(key), "logprob": -float(key)}
                                 for key in range(1, 33)],
            }]
        rows.append(row)
    rows.append({"stop": True, "prompt": "identical", "tokens_evaluated": 7,
                 "tokens_predicted": len(tokens), "generation_settings": {
                     "n_probs": 32, "post_sampling_probs": False,
                     "backend_sampling": False, "seed": 42, "temperature": 0.0}})
    path.write_text("\n".join("data: " + json.dumps(row) for row in rows))


class ProbabilityComparisonTests(unittest.TestCase):
    def test_only_first_divergent_prediction_has_a_comparable_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            full, reduced = Path(directory) / "full", Path(directory) / "reduced"
            stream(full, [1, 1, 1])
            stream(reduced, [1, 2, 3])
            result = compare(full, reduced, {"tokens": [1, 1, 1], "request_index": 50},
                             {"tokens": [1, 2, 3]})
        self.assertEqual(result["same_prefix_comparisons"], 2)
        self.assertEqual(result["excluded_after_divergence"], 1)
        self.assertEqual(result["first_divergence"]["output_position"], 2)
        self.assertEqual(result["first_divergence"]["context_tokens"], 8)

    def test_identical_streams_use_all_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            full, reduced = Path(directory) / "full", Path(directory) / "reduced"
            stream(full, [1, 2, 3])
            stream(reduced, [1, 2, 3])
            result = compare(full, reduced, {"tokens": [1, 2, 3], "request_index": 50},
                             {"tokens": [1, 2, 3]})
        self.assertTrue(result["identical"])
        self.assertEqual(result["same_prefix_comparisons"], 3)
        self.assertEqual(result["maximum_same_prefix_shared_logprob_delta"], 0)

    def test_missing_diagnostics_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream"
            stream(path, [1], probabilities=False)
            with self.assertRaisesRegex(AssertionError, "missing diagnostic observations"):
                read_stream(path)


if __name__ == "__main__":
    unittest.main()
