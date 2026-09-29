"""Summarize the three saved cancellation-prerequisite receipts, without devices."""

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
PHYSICAL = ROOT / "physical"


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize_case(case, boot_id):
    root = PHYSICAL / "results" / case
    result = read_json(root / "RESULT.json")
    assert result["status"] == "passed" and result["boot_id"] == boot_id
    assert result["dma_settled"] and result["usb_restored"]
    text = (root / "service.events").read_text()
    events = re.findall(r"^(\d+) pid=(\d+) stage=(.*?) index=(-?\d+)$", text, re.M)
    assert events and len({pid for _, pid, _, _ in events}) == 1

    def times(stage):
        return [int(at) for at, _pid, name, _index in events if name == stage]

    def interval(first, last):
        return (max(times(last)) - min(times(first))) / 1e6

    observed = read_json(root / "observation.json")
    assert observed["seconds"] >= 60
    assert all(sample["identity"].splitlines() == [boot_id, "ptp,adb", "ptp,adb"]
               for sample in observed["samples"])
    allocations = (root / "live_allocations.txt").read_text().splitlines()
    released = (root / "released_allocations.txt").read_text().splitlines()
    inodes = [row.split(":")[0] for row in allocations]
    assert len(set(inodes)) == 8
    assert released == [inode + ":absent" for inode in inodes]
    summary = {
        "status": "passed",
        "boot_id": boot_id,
        "worker_pid": int(events[0][1]),
        "dma_allocations": len(allocations),
        "dma_bytes": sum(int(row.split(":")[1]) for row in allocations),
        "dma_allocations_remaining": 0,
        "drain_ms": interval("drain_begin", "drain_complete"),
        "restore_ack_wait_ms": interval("restore_wait_begin", "restore_confirmed"),
        "drain_to_buffers_released_ms": interval("drain_begin", "buffers_release_done"),
        "post_cleanup_observation_seconds": observed["seconds"],
        "model_compute_tested": result["model_compute_tested"],
        "result_sha256": sha256(root / "RESULT.json"),
        "events_sha256": sha256(root / "service.events"),
    }
    if case == "abort":
        assert result["forced_cancellation_qualified"]
        assert len(times("cancel_pending_rx_verified")) == 4
        statuses = [(int(index), int(name.removeprefix("cancel_fence_status_")))
                    for _at, _pid, name, index in events
                    if name.startswith("cancel_fence_status_")]
        assert statuses == [(i, -104 if i % 2 == 0 else 1) for i in range(8)]
        summary.update({
            "receive_statuses": [status for index, status in statuses if index % 2 == 0],
            "unused_transmit_fence_statuses": [status for index, status in statuses if index % 2],
            "detach_ms": interval("cancel_detach_begin", "cancel_detach_done"),
            "post_detach_fence_wait_ms": interval("attached_fence_wait_begin", "attached_fence_wait_done"),
            "idle_receive_cancellation_only": True,
            "host_bulk_submissions": read_json(root / "CANCELLATION_REQUEST.json")["host_bulk_submissions"],
        })
    return summary


def main():
    boot = read_json(PHYSICAL / "CANDIDATE_BOOT_RESULT.json")
    assert boot["status"] == "CANDIDATE_BOOTED_IDENTITY_VERIFIED"
    assert boot["partition_flash_count"] == 0
    record = {
        "schema": "s42-idle-rx-cancellation-prerequisite-v1",
        "status": "PASS",
        "boot": boot,
        "cases": {case: summarize_case(case, boot["boot_id"])
                  for case in ("smoke", "stop", "abort")},
        "scope": "Four pending 4096-byte USB receives; no HTP compute cancellation.",
        "ffn_relocation_executed": False,
        "old_kernel_qualification_reused": False,
        "boot_result_sha256": sha256(PHYSICAL / "CANDIDATE_BOOT_RESULT.json"),
    }
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
