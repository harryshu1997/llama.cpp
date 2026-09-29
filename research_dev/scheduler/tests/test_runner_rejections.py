"""The BurstGPT runner records submission rejections instead of aborting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from research_dev.scheduler.adapters.coordinator import RuntimeSubmissionRejection
from research_dev.scheduler.campaigns.burstgpt import runner


def _row(event_id, index, input_tokens, output_tokens, arrival_us):
    return {
        "combined_index": index,
        "model_id": "gemma-model",
        "row": {
            "event_id": event_id,
            "arrival_us": arrival_us,
            "slo_us": 30_000_000,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prompt_tokens": list(range(input_tokens)),
        },
    }


class RunnerRejectionTests(unittest.TestCase):
    def test_screen_request_shapes_reports_every_request(self) -> None:
        class Scheduler:
            def request_shape_support(self, request, model_id):
                tokens = request.input_tokens + request.output_tokens
                return SimpleNamespace(to_json=lambda: {
                    "request_id": request.request_id, "model_id": model_id, "tokens": tokens,
                    "supported": tokens <= 2048,
                    "reason": None if tokens <= 2048 else "REQUEST_EXCEEDS_CONTEXT_CAPACITY",
                })

        merged = [_row("a", 48, 309, 11, 851_000_000), _row("b", 49, 1884, 491, 876_000_000)]
        screen = runner._screen_request_shapes(Scheduler(), merged)
        self.assertEqual(screen["checked"], 2)
        self.assertEqual([row["request_id"] for row in screen["unsupported"]], ["b"])
        self.assertEqual(screen["unsupported"][0]["combined_request_index"], 49)
        self.assertEqual(screen["verdicts"][0]["supported"], True)

    def test_submit_arrivals_records_rejection_and_continues(self) -> None:
        submitted = []

        class Coordinator:
            def __init__(self):
                self._rejections = []

            def wait_for_arrival(self, arrival_us):
                return arrival_us

            def observed_at_us(self):
                return 900_000_000

            def submit(self, submission, *, observed_at_us):
                submitted.append(submission.request.request_id)
                if submission.request.request_id == "b":
                    self._rejections.append(RuntimeSubmissionRejection(
                        request_id="b", model_id=submission.model_id,
                        arrival_us=submission.request.arrival_us, observed_at_us=observed_at_us,
                        reason="REQUEST_EXCEEDS_CONTEXT_CAPACITY",
                        details={"tokens": 2375, "maximum_capacity_tokens": 2048},
                    ))
                    return None
                return SimpleNamespace(ticket_id=submission.request.request_id)

            def rejections(self):
                return tuple(self._rejections)

        class Rig:
            def snapshot(self, request, model_id, captured_at_us):
                return SimpleNamespace(to_json=lambda: {"snapshot": request.request_id})

        merged = [_row("a", 48, 309, 11, 1), _row("b", 49, 1884, 491, 2), _row("c", 52, 303, 14, 3)]
        state = runner._ArrivalState()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "CanonicalRuntimeSubmission", lambda **kw: SimpleNamespace(**kw)
        ):
            root = Path(directory)
            (root / "streams").mkdir(); (root / "snapshots").mkdir()
            runner._submit_arrivals(
                SimpleNamespace(selection_mode="energy-aware"), merged, Coordinator(), Rig(),
                0, root / "streams", root / "snapshots", {"gemma-model": "gemma-alias"}, state,
            )
        self.assertEqual(submitted, ["a", "b", "c"])
        self.assertEqual(list(state.rejections), ["b"])
        rejection = state.rejections["b"]
        self.assertEqual(rejection["reason"], "REQUEST_EXCEEDS_CONTEXT_CAPACITY")
        self.assertEqual(rejection["combined_request_index"], 49)
        self.assertEqual(rejection["input_tokens"], 1884)
        self.assertEqual(rejection["details"]["tokens"], 2375)
        self.assertEqual(set(state.overheads), {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
