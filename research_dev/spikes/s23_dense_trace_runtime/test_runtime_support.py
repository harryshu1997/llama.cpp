#!/usr/bin/env python3

from __future__ import annotations

import threading
import time
import unittest

from runtime_support import SequenceSlotPool, SerializedStageClient, SlotError


class SequenceSlotPoolTests(unittest.TestCase):
    def test_slots_are_reused_only_after_exact_release(self) -> None:
        pool = SequenceSlotPool(2)
        self.assertEqual(pool.try_acquire(10), 0)
        self.assertEqual(pool.try_acquire(11), 1)
        self.assertIsNone(pool.try_acquire(12))
        pool.release(0, 10)
        self.assertEqual(pool.try_acquire(12), 0)

    def test_wrong_owner_and_double_release_fail(self) -> None:
        pool = SequenceSlotPool(1)
        self.assertEqual(pool.try_acquire(10), 0)
        with self.assertRaisesRegex(SlotError, "ownership"):
            pool.release(0, 11)
        pool.release(0, 10)
        with self.assertRaisesRegex(SlotError, "ownership"):
            pool.release(0, 10)

    def test_duplicate_request_lease_fails(self) -> None:
        pool = SequenceSlotPool(2)
        pool.try_acquire(10)
        with self.assertRaisesRegex(SlotError, "already owns"):
            pool.try_acquire(10)


class FakeClient:
    def __init__(self) -> None:
        self.active = 0
        self.overlap = False

    def batch(self, _rows):
        self.active += 1
        if self.active != 1:
            self.overlap = True
        time.sleep(0.01)
        self.active -= 1
        return ()

    def remove(self, *_args):
        self.active += 1
        if self.active != 1:
            self.overlap = True
        time.sleep(0.01)
        self.active -= 1
        return None


class SerializedClientTests(unittest.TestCase):
    def test_batch_and_remove_do_not_overlap_on_one_socket(self) -> None:
        raw = FakeClient()
        client = SerializedStageClient(raw)
        first = threading.Thread(target=client.batch, args=([],))
        second = threading.Thread(target=client.remove, args=(0, 1, 1))
        first.start()
        second.start()
        first.join()
        second.join()
        self.assertFalse(raw.overlap)


if __name__ == "__main__":
    unittest.main()
