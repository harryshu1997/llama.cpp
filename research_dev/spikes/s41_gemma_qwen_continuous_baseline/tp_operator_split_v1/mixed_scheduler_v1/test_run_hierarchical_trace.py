#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_hierarchical_trace


class SleepUntilTest(unittest.TestCase):
    def test_deadline_crossed_before_sleep_is_not_negative(self) -> None:
        clock = iter((90, 101))
        delays = []

        with mock.patch.object(
            run_hierarchical_trace.time,
            "monotonic_ns",
            side_effect=lambda: next(clock),
        ), mock.patch.object(
            run_hierarchical_trace.time,
            "sleep",
            side_effect=delays.append,
        ):
            run_hierarchical_trace.sleep_until_ns(100)

        self.assertEqual(delays, [10 / 1e9])
        self.assertGreaterEqual(delays[0], 0)

    def test_elapsed_deadline_does_not_sleep(self) -> None:
        with mock.patch.object(
            run_hierarchical_trace.time,
            "monotonic_ns",
            return_value=101,
        ), mock.patch.object(run_hierarchical_trace.time, "sleep") as sleep:
            run_hierarchical_trace.sleep_until_ns(100)

        sleep.assert_not_called()

    def test_resident_release_requires_a_completed_overlay_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "release.json"
            receipt = {
                "released_at_ns": 100,
                "schema": "s42-resident-release-v1",
                "status": "OVERLAY_COMPLETE",
            }
            path.write_text(json.dumps(receipt), encoding="ascii")
            self.assertEqual(
                run_hierarchical_trace.wait_for_resident_release(path, 1),
                receipt,
            )

            path.write_text(json.dumps({
                **receipt,
                "status": "CONTROLLER_ABORT",
            }), encoding="ascii")
            with self.assertRaisesRegex(
                run_hierarchical_trace.run_trace.RunError,
                "resident release receipt",
            ):
                run_hierarchical_trace.wait_for_resident_release(path, 1)


if __name__ == "__main__":
    unittest.main()
