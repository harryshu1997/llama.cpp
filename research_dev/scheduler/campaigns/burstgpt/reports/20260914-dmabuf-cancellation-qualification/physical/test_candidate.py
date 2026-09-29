import unittest
from pathlib import Path
from src import run_direct_usb_device as device


def events():
    rows = [("request_or_setup_failed", -1), ("request_error:direct USB interrupted; cancel and drain required", -1)]
    rows += [("cancel_pending_rx_verified", i) for i in range(4)]
    rows += [("attached_fence_export_done", i) for i in range(8)]
    for i in reversed(range(8)):
        rows += [("cancel_detach_begin", i), ("cancel_detach_done", i)]
    for i in range(8):
        rows += [("cancel_fence_status_" + ("-104" if i % 2 == 0 else "1"), i), ("attached_fence_wait_done", i)]
    rows += [(name, -1) for name in ("attached_fences_release_done", "drain_complete", "restore_confirmed", "in_close_begin", "out_close_begin", "ep0_close_begin", "buffers_release_done")]
    return rows


def encode(rows):
    return "\n".join(f"{n} pid=100 stage={stage} index={index}" for n, (stage, index) in enumerate(rows))


class CandidateTests(unittest.TestCase):
    def test_complete_idle_cancellation(self):
        device.validate_lifecycle_events(encode(events()), cancellation=True)

    def test_missing_or_repeated_completion_is_rejected(self):
        rows = events()
        for n in range(len(rows)):
            with self.subTest(n=n), self.assertRaises(RuntimeError):
                device.validate_lifecycle_events(encode(rows[:n] + rows[n + 1:]), cancellation=True)
        with self.assertRaises(RuntimeError):
            device.validate_lifecycle_events(encode(rows + [rows[-1]]), cancellation=True)

    def test_completed_receive_is_not_cancellation(self):
        rows = [("cancel_fence_status_1" if s == "cancel_fence_status_-104" else s, i) for s, i in events()]
        with self.assertRaises(RuntimeError):
            device.validate_lifecycle_events(encode(rows), cancellation=True)

    def test_release_before_completion_is_rejected(self):
        rows = events()
        release = next(row for row in rows if row[0] == "attached_fences_release_done")
        rows.remove(release)
        rows.insert(0, release)
        with self.assertRaises(RuntimeError):
            device.validate_lifecycle_events(encode(rows), cancellation=True)

    def test_candidate_hashes_are_exact_and_old_bundle_is_refused(self):
        root = Path(__file__).resolve().parent
        device.bundle_hashes(root / "candidate-bundle")
        with self.assertRaises(RuntimeError):
            device.bundle_hashes(root / "reference-bundle")


if __name__ == "__main__":
    unittest.main()
