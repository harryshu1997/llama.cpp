#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest

import validate_report


class PersistentHostTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(validate_report.REPORT.read_text(encoding="ascii"))

    def assert_rejected(self, mutate) -> None:
        value = copy.deepcopy(self.report)
        mutate(value)
        with self.assertRaises(validate_report.ValidationError):
            validate_report.validate(value)

    def test_frozen_report_passes(self) -> None:
        validate_report.validate(copy.deepcopy(self.report))

    def test_host_pid_change_rejected(self) -> None:
        self.assert_rejected(lambda value: value.__setitem__("host_pid", 1))

    def test_worker_pid_change_rejected(self) -> None:
        self.assert_rejected(lambda value: value.__setitem__("resident_worker_pid", 1))

    def test_missing_session_rejected(self) -> None:
        self.assert_rejected(lambda value: value["sessions"].pop())

    def test_wrong_detach_reset_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["sessions"][0]["phone_session"].__setitem__(
                "reset_applied", False,
            )
        )

    def test_noncontiguous_steps_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["sessions"][1]["phone_session"].__setitem__(
                "steps_total", 769,
            )
        )

    def test_host_fallback_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["sessions"][0]["host_placement"][
                "compute_by_buffer_type"
            ].__setitem__("CPU", 1)
        )

    def test_token_change_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["sessions"][0]["result"]["token_ids"][0].__setitem__(
                0, 0,
            )
        )

    def test_latency_over_budget_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["sessions"][0]["result"].__setitem__(
                "elapsed_us", 4_000_001,
            )
        )


if __name__ == "__main__":
    unittest.main()
