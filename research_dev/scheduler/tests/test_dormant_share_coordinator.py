"""Rig-level admission for the decode-only FFN relocation (mechanics, no devices)."""
from __future__ import annotations

import threading
import time
import unittest

from research_dev.scheduler._internal.decode_split_selection import ShareBinding
from research_dev.scheduler.adapters.dormant_share_coordinator import DormantShareAdmissionError, DormantShareCoordinator

GIB = 1024 ** 3
QWEN = "sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718"
MASK = (1 << 18) - 1
EP = "http://127.0.0.1:18571"


def ack(generation, released, host_columns=0):
    return {"calls": 12, "dormant_host_share": True, "dormant_layer_mask": MASK, "dormant_host_columns": host_columns,
            "dormant_release_generation": generation, "dormant_released_bytes": released, "dormant_release_elapsed_us": 90000}


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.coordinator = DormantShareCoordinator(budget_bytes=20 * GIB, safety_bytes=0, hold_timeout_s=5.0)
        self.binding = ShareBinding(EP, QWEN, MASK, 0, 9 * GIB, column_quantum=2176)

    def test_booking_credit_and_prompt_gate(self):
        self.coordinator.book_server(EP, base_bytes=7 * GIB, workspace_bytes=GIB, binding=self.binding)
        self.assertEqual(self.coordinator.reserved_bytes(), 17 * GIB)
        self.assertIsNone(self.coordinator.before_prompt(EP, "req-1"))  # share resident: nothing to wait for
        # acknowledgements without a release, or repeating a generation, credit nothing
        self.assertIsNone(self.coordinator.on_control_ack(EP, "req-1", {"calls": 3}))
        self.assertEqual(self.coordinator.on_control_ack(EP, "req-1", ack(1, 9 * GIB)), 9 * GIB)
        self.assertIsNone(self.coordinator.on_control_ack(EP, "req-1", ack(1, 9 * GIB)))
        self.assertEqual((self.coordinator.reserved_bytes(), self.coordinator.state(EP)), (8 * GIB, "decode-released"))
        # a tenant takes the room; the next prompt on this server must wait
        self.coordinator.reserve_growth("tenant", 8 * GIB)
        held = []

        def prompt():
            held.append(self.coordinator.before_prompt(EP, "req-2"))
        thread = threading.Thread(target=prompt)
        thread.start()
        time.sleep(0.3)
        self.assertTrue(thread.is_alive())
        kinds = [row["kind"] for row in self.coordinator.events()]
        self.assertIn("prompt_held", kinds)
        self.coordinator.release_growth("tenant")
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(held[0]["request_id"], "req-2")
        self.assertGreaterEqual(held[0]["waited_s"], 0.25)
        self.assertEqual((self.coordinator.reserved_bytes(), self.coordinator.state(EP)), (17 * GIB, "prefill-resident"))
        # the second generation credits again (a smaller release than booked keeps the remainder charged)
        self.assertEqual(self.coordinator.on_control_ack(EP, "req-2", ack(2, 4 * GIB, host_columns=8704)), 4 * GIB)
        self.assertEqual(self.coordinator.reserved_bytes(), 13 * GIB)
        # a subset of the booked layers (a helper attached to layers 0-5 only) is a valid smaller release
        subset = dict(ack(3, 3 * GIB)); subset["dormant_layer_mask"] = 63
        self.coordinator.forget_server(EP)
        self.coordinator.book_server(EP, base_bytes=7 * GIB, workspace_bytes=GIB, binding=self.binding)
        self.assertEqual(self.coordinator.on_control_ack(EP, "req-3", subset), 3 * GIB)
        # layers outside the booking are refused as an accounting event, never an exception
        outside = dict(ack(4, GIB)); outside["dormant_layer_mask"] = 1 << 30
        before = self.coordinator.reserved_bytes()
        self.assertIsNone(self.coordinator.on_control_ack(EP, "req-4", outside))
        self.assertEqual(self.coordinator.reserved_bytes(), before)
        self.assertIn("release_credit_refused", [row["kind"] for row in self.coordinator.events()])
        self.coordinator.forget_server(EP)
        self.assertEqual(self.coordinator.reserved_bytes(), 0)

    def test_hold_times_out_fail_closed(self):
        coordinator = DormantShareCoordinator(budget_bytes=20 * GIB, safety_bytes=0, hold_timeout_s=0.5)
        coordinator.book_server(EP, base_bytes=7 * GIB, binding=self.binding)
        coordinator.on_control_ack(EP, "req-1", ack(1, 9 * GIB))
        coordinator.reserve_growth("tenant", 10 * GIB)
        with self.assertRaises(DormantShareAdmissionError):
            coordinator.before_prompt(EP, "req-2")
        self.assertEqual(coordinator.state(EP), "restore-blocked")
        self.assertIn("prompt_hold_timeout", [row["kind"] for row in coordinator.events()])

    def test_budget_refuses_a_server_that_does_not_fit(self):
        self.coordinator.book_server("http://127.0.0.1:1", base_bytes=12 * GIB)
        with self.assertRaises(DormantShareAdmissionError):
            self.coordinator.book_server(EP, base_bytes=7 * GIB, workspace_bytes=GIB, binding=self.binding)
        self.assertEqual(self.coordinator.reserved_bytes(), 12 * GIB)  # nothing partially booked
        self.assertIsNone(self.coordinator.booking(EP))
        self.coordinator.book_server("http://127.0.0.1:2", base_bytes=3 * GIB)  # a server without a share needs no binding
        self.assertIsNone(self.coordinator.before_prompt("http://127.0.0.1:2", "r"))
        with self.assertRaises(DormantShareAdmissionError):
            self.coordinator.book_server("http://127.0.0.1:2", base_bytes=GIB)

    def test_non_strict_booking_records_over_subscription_and_books_what_fits(self):
        self.coordinator.book_server("http://127.0.0.1:1", base_bytes=15 * GIB)
        self.coordinator.book_server(EP, base_bytes=7 * GIB, workspace_bytes=GIB, binding=self.binding, strict=False)
        booking = self.coordinator.booking(EP)
        self.assertEqual((booking.base_bytes, booking.binding), (5 * GIB, None))
        self.assertEqual(self.coordinator.reserved_bytes(), 20 * GIB)
        self.assertIn("server_booking_over_budget", [row["kind"] for row in self.coordinator.events()])
        self.assertIsNone(self.coordinator.before_prompt(EP, "r"))  # no share booked: nothing to gate
        self.coordinator.forget_server(EP)
        self.assertEqual(self.coordinator.reserved_bytes(), 15 * GIB)

    def test_relaunch_on_the_same_endpoint_restarts_generations(self):
        self.coordinator.book_server(EP, base_bytes=7 * GIB, binding=self.binding)
        self.assertEqual(self.coordinator.on_control_ack(EP, "r1", ack(1, 3 * GIB)), 3 * GIB)
        self.coordinator.forget_server(EP)
        self.coordinator.book_server(EP, base_bytes=7 * GIB, binding=self.binding)  # a new server, same endpoint
        self.assertEqual(self.coordinator.on_control_ack(EP, "r2", ack(1, 3 * GIB)), 3 * GIB)
        self.assertNotIn("release_credit_refused", [row["kind"] for row in self.coordinator.events()])

    def test_ack_from_an_unbooked_or_shareless_server_is_ignored(self):
        self.coordinator.book_server("http://127.0.0.1:2", base_bytes=3 * GIB)
        self.assertIsNone(self.coordinator.on_control_ack("http://127.0.0.1:2", "r", ack(1, GIB)))
        self.assertIsNone(self.coordinator.on_control_ack("http://127.0.0.1:9", "r", ack(1, GIB)))


if __name__ == "__main__":
    unittest.main()
