"""2026-10-08: placement failures map to their physical cause, not the PLACEMENT_INFEASIBLE fallback.

The Qwen3-32B Q8_0 two-phone arm (pA1q8a) labelled every co-helper offload route PLACEMENT_INFEASIBLE although the
planner said ``no feasible placement at operator layer:21:ffn: candidate_gate=0, memory=0, transfer=1, deadline=0``:
the parser split on ``": candidate_gate="`` and lost that key, so the first item ``"0"`` raised and fell back.
"""

from __future__ import annotations

import unittest

from research_dev.scheduler._internal.route_generation import _placement_rejection_reasons


class PlacementRejectionReasonTests(unittest.TestCase):
    def test_counts_map_to_their_causes(self) -> None:
        message = "no feasible placement at operator layer:21:ffn: candidate_gate={}, memory={}, transfer={}, deadline={}"
        cases = (
            ((0, 0, 1, 0), ("TRANSFER_PLAN_ABSENT",)),
            ((0, 2, 0, 0), ("MEMORY_CAPACITY",)),
            ((1, 0, 0, 3), ("KERNEL_SUPPORT_ABSENT", "SLO_UPPER_BOUND")),
            ((0, 0, 0, 0), ("PLACEMENT_INFEASIBLE",)),
        )
        for counts, expected in cases:
            with self.subTest(counts=counts):
                self.assertEqual(_placement_rejection_reasons(message.format(*counts)), expected)

    def test_other_messages(self) -> None:
        self.assertEqual(_placement_rejection_reasons("no placement meets the final transfer to desktop-cpu"),
                         ("TRANSFER_PLAN_ABSENT",))
        self.assertEqual(_placement_rejection_reasons("planner gave up"), ("PLACEMENT_INFEASIBLE",))
        self.assertEqual(_placement_rejection_reasons("x: candidate_gate=zero"), ("PLACEMENT_INFEASIBLE",))


if __name__ == "__main__":
    unittest.main()
