"""Per-process CPU, aggregate jiffies and CPU frequency alongside RAPL samples."""

import os
from pathlib import Path
import tempfile
import time
import unittest

from research_dev.scheduler.adapters.host_runtime import (
    HOST_ACTIVITY_RSS_ALWAYS_BYTES,
    HostEnergySampler,
    HostMetricCallbacks,
    _filter_host_activity,
    linux_host_activity,
)


def fake_proc(root: Path, *, jiffies: str, processes: dict[int, tuple[str, int, int, int]], khz=(2_400_000, 800_000)) -> None:
    proc = root / "proc"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "stat").write_text("cpu  " + jiffies + "\ncpu0 1 2 3 4 5 6 7 0 0 0\n")
    for pid, (comm, utime, stime, rss_pages) in processes.items():
        (proc / str(pid)).mkdir(exist_ok=True)
        fields = ["S", "1", "1", "1", "0", "-1", "4194560", "0", "0", "0", "0",
                  str(utime), str(stime), "0", "0", "20", "0", "1", "0", "100", "4096",
                  str(rss_pages), "0"]
        (proc / str(pid) / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields) + "\n")
    (proc / "not-a-pid").mkdir(exist_ok=True)
    # Threads of pid 7: the main thread and a named worker.
    for tid, (comm, utime, stime) in {7: ("llama-server", 100, 20), 8: ("request-helper-", 20, 10)}.items():
        (proc / "7" / "task" / str(tid)).mkdir(parents=True, exist_ok=True)
        fields = ["S", "1", "1", "1", "0", "-1", "4194560", "0", "0", "0", "0",
                  str(utime), str(stime), "0", "0", "20", "0", "1", "0", str(500 + tid), "4096", "10", "0"]
        (proc / "7" / "task" / str(tid) / "stat").write_text(f"{tid} ({comm}) " + " ".join(fields) + "\n")
    cpu = root / "cpu"
    for index, value in enumerate(khz):
        (cpu / f"cpu{index}" / "cpufreq").mkdir(parents=True, exist_ok=True)
        (cpu / f"cpu{index}" / "cpufreq" / "scaling_cur_freq").write_text(f"{value}\n")


class HostActivityTests(unittest.TestCase):
    def test_reads_raw_counters_including_awkward_command_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_proc(root, jiffies="571387 488060 135644 86672890 230260 0 28403 0 0 0",
                      processes={7: ("llama-server", 120, 30, 6_000_000), 9: ("irq/145-iwlwifi", 5, 0, 0),
                                 11: ("weird (name) x", 1, 1, 10)})
            activity = linux_host_activity(root / "proc", root / "cpu", thread_pids=(7,))
        self.assertEqual([(t["tid"], t["comm"], t["cpu_ticks"], t["start_ticks"]) for t in activity["threads"]],
                         [(7, "llama-server", 120, 507), (8, "request-helper-", 30, 508)])
        self.assertEqual(activity["processes"][0]["start_ticks"], 100)
        self.assertEqual(activity["cpu_jiffies"], {
            "user": 571387, "nice": 488060, "system": 135644, "idle": 86672890, "iowait": 230260,
            "irq": 0, "softirq": 28403, "steal": 0, "guest": 0, "guest_nice": 0})
        self.assertEqual(activity["cpu_khz"], {"count": 2, "max": 2_400_000, "mean": 1_600_000, "min": 800_000})
        self.assertGreater(activity["clock_ticks_per_s"], 0)
        rows = {row["pid"]: row for row in activity["processes"]}
        self.assertEqual(sorted(rows), [7, 9, 11])
        self.assertEqual(rows[7]["cpu_ticks"], 150)
        self.assertEqual(rows[7]["comm"], "llama-server")
        self.assertEqual(rows[7]["rss_bytes"], 6_000_000 * 4096)
        self.assertEqual(rows[11]["comm"], "weird (name) x")
        self.assertEqual(rows[11]["cpu_ticks"], 2)
        self.assertGreaterEqual(activity["scan_ns"], 0)

    def test_missing_cpufreq_is_reported_as_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_proc(root, jiffies="1 2 3 4 5 6 7 0 0 0", processes={}, khz=())
            self.assertIsNone(linux_host_activity(root / "proc", root / "cpu", thread_pids=())["cpu_khz"])

    def test_filter_keeps_changed_ticks_and_large_rss_only(self) -> None:
        previous: dict[int, int] = {}
        big = HOST_ACTIVITY_RSS_ALWAYS_BYTES
        first = _filter_host_activity({"processes": [
            {"pid": 1, "comm": "a", "cpu_ticks": 10, "rss_bytes": 0},
            {"pid": 2, "comm": "b", "cpu_ticks": 20, "rss_bytes": big},
            {"pid": 3, "comm": "c", "cpu_ticks": 0, "rss_bytes": 0},
        ], "cpu_jiffies": {}}, previous)
        # Everything is new on the first sample.
        self.assertEqual([row["pid"] for row in first["processes"]], [1, 2, 3])
        second = _filter_host_activity({"processes": [
            {"pid": 1, "comm": "a", "cpu_ticks": 10, "rss_bytes": 0},     # idle: dropped
            {"pid": 2, "comm": "b", "cpu_ticks": 20, "rss_bytes": big},   # idle but large RSS: kept
            {"pid": 3, "comm": "c", "cpu_ticks": 4, "rss_bytes": 0},      # used CPU: kept
        ], "cpu_jiffies": {}}, previous)
        self.assertEqual([row["pid"] for row in second["processes"]], [2, 3])
        self.assertEqual(previous, {1: 10, 2: 20, 3: 4})
        # A process that exited is forgotten; a returning pid counts as new.
        third = _filter_host_activity({"processes": [
            {"pid": 3, "comm": "c", "cpu_ticks": 4, "rss_bytes": 0},
        ], "cpu_jiffies": {}}, previous)
        self.assertEqual(third["processes"], [])
        self.assertEqual(previous, {3: 4})
        # Threads are filtered the same way, keyed by (pid, tid).
        previous = {}
        first = _filter_host_activity({"processes": [], "threads": [
            {"pid": 9, "tid": 9, "comm": "python3", "cpu_ticks": 5, "start_ticks": 1},
            {"pid": 9, "tid": 10, "comm": "request-helper-", "cpu_ticks": 0, "start_ticks": 2},
        ]}, previous)
        self.assertEqual([t["tid"] for t in first["threads"]], [9, 10])
        second = _filter_host_activity({"processes": [], "threads": [
            {"pid": 9, "tid": 9, "comm": "python3", "cpu_ticks": 5, "start_ticks": 1},
            {"pid": 9, "tid": 10, "comm": "request-helper-", "cpu_ticks": 40, "start_ticks": 2},
        ]}, previous)
        self.assertEqual([t["tid"] for t in second["threads"]], [10])
        self.assertEqual(previous[(9, 10)], 40)

    def test_sampler_rows_carry_host_activity_and_survive_its_errors(self) -> None:
        calls = 0

        def activity():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("proc vanished")
            return {"clock_ticks_per_s": 100, "cpu_jiffies": {"user": calls}, "cpu_khz": None,
                    "processes": [{"pid": 1, "comm": "x", "cpu_ticks": calls, "rss_bytes": 0}],
                    "sample_t_ns": time.monotonic_ns(), "scan_ns": 1}

        sampler = HostEnergySampler(
            HostMetricCallbacks(
                gpu_snapshot=lambda: {"power_mw": 1000},
                rapl_package_snapshot=lambda: {"energy_uj": 1, "max_energy_range_uj": 1000,
                                               "sample_t_ns": time.monotonic_ns()},
                system_memory=lambda: {"available_bytes": 1},
                host_activity=activity,
            ),
            interval_s=0.01,
        )
        sampler.start()
        deadline = time.monotonic() + 2
        while len(sampler.rows()) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        sampler.stop()
        rows = sampler.rows()
        self.assertGreaterEqual(len(rows), 4)
        self.assertIn("host_activity", rows[0])
        self.assertEqual(rows[0]["host_activity"]["processes"][0]["cpu_ticks"], 1)
        # The failed probe still produced a row, without host activity.
        self.assertNotIn("host_activity", rows[1])
        self.assertIn("host_activity", rows[2])
        diagnostics = sampler.diagnostics()
        self.assertEqual(diagnostics["events"][0]["source"], "host_activity")
        self.assertIsNone(diagnostics["fatal_error"])
        # Rows without the callback keep the old shape.
        plain = HostEnergySampler(HostMetricCallbacks(
            gpu_snapshot=lambda: {"power_mw": 1}, rapl_package_snapshot=lambda: {
                "energy_uj": 1, "max_energy_range_uj": 10, "sample_t_ns": time.monotonic_ns()},
            system_memory=lambda: {"available_bytes": 1}), interval_s=0.01)
        plain.start()
        deadline = time.monotonic() + 1
        while len(plain.rows()) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        plain.stop()
        self.assertNotIn("host_activity", plain.rows()[0])

    def test_real_proc_is_readable_when_present(self) -> None:
        if not Path("/proc/stat").exists():
            self.skipTest("no procfs")
        activity = linux_host_activity(thread_pids=(os.getpid(),))
        self.assertIn("user", activity["cpu_jiffies"])
        self.assertTrue(any(row["cpu_ticks"] > 0 for row in activity["processes"]))
        self.assertTrue(any(row["tid"] == os.getpid() for row in activity["threads"]))


if __name__ == "__main__":
    unittest.main()
