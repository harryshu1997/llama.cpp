"""Host gates of the portable path: other users' load and the serving GPU.

``ForeignLoadWatch`` samples how many cores run work that is not this run (total busy time of the host
minus the run's own process tree). A metered phase waits for a quiet host, is cancelled when foreign load
trips the watch, and is repeated when its mean excess was too high: on a shared host this is what lets a
run finish between other users' bursts without touching their processes.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable, Sequence

from ..adapters.contracts import PhysicalAdapterError


def host_busy_jiffies(stat_path: str = "/proc/stat") -> int:
    """Busy time of every CPU: everything but idle and iowait."""
    fields = [int(value) for value in Path(stat_path).read_text(encoding="ascii").splitlines()[0].split()[1:]]
    return sum(fields) - fields[3] - fields[4]


def process_tree_jiffies(roots: Sequence[int], proc: str = "/proc") -> int:
    """utime + stime (+ reaped children) of the given processes and all their live descendants."""
    table: dict[int, tuple[int, int]] = {}
    for entry in os.scandir(proc):
        if not entry.name.isdigit():
            continue
        try:
            fields = Path(proc, entry.name, "stat").read_text(encoding="ascii", errors="replace") \
                .rsplit(")", 1)[1].split()
            table[int(entry.name)] = (int(fields[1]),
                                      int(fields[11]) + int(fields[12]) + int(fields[13]) + int(fields[14]))
        except (OSError, IndexError, ValueError):
            continue
    own = set(roots)
    grew = True
    while grew:
        grew = False
        for pid, (parent, _) in table.items():
            if parent in own and pid not in own:
                own.add(pid)
                grew = True
    return sum(table[pid][1] for pid in own if pid in table)


class ForeignLoadWatch(threading.Thread):
    """Cores of foreign work, sampled every ``period_s``.

    More than ``limit_cores`` in every sample of the last ``window_s`` sets ``tripped`` and calls
    ``on_trip`` once (until ``rearm``). Every sample is kept as ``(monotonic_ns, cores)``.
    """

    def __init__(self, own_pids: Callable[[], Sequence[int]], *, limit_cores: float, window_s: float,
                 on_trip: Callable[[], None], period_s: float = 5.0,
                 busy: Callable[[], int] = host_busy_jiffies,
                 own: Callable[[Sequence[int]], int] = process_tree_jiffies,
                 clock: Callable[[], float] = time.monotonic, clock_ns: Callable[[], int] = time.monotonic_ns,
                 ticks_per_s: int | None = None) -> None:
        super().__init__(daemon=True)
        if limit_cores <= 0 or window_s <= 0 or period_s <= 0:
            raise PhysicalAdapterError("foreign load watch needs positive limit, window and period")
        self.own_pids, self.limit_cores, self.window_s, self.on_trip = own_pids, limit_cores, window_s, on_trip
        self.period_s = period_s
        self._busy, self._own, self._clock, self._clock_ns = busy, own, clock, clock_ns
        self._hz = ticks_per_s or os.sysconf("SC_CLK_TCK")
        self.samples: list[tuple[int, float]] = []
        self.tripped: dict[str, object] | None = None
        self.stop_event = threading.Event()
        self._last: tuple[float, int, int] | None = None

    def sample(self) -> float | None:
        """Take one sample now (the thread's loop body); None on the first call."""
        now, busy, own = self._clock(), self._busy(), self._own(self.own_pids())
        previous, self._last = self._last, (now, busy, own)
        if previous is None or now <= previous[0]:
            return None
        cores = max(0.0, ((busy - previous[1]) - (own - previous[2])) / self._hz / (now - previous[0]))
        self.samples.append((self._clock_ns(), round(cores, 2)))
        count = max(1, int(self.window_s / self.period_s))
        recent = [value for _, value in self.samples[-count:]]
        if self.tripped is None and len(recent) == count and min(recent) > self.limit_cores:
            self.tripped = {"at_ns": self._clock_ns(), "cores": recent}
            self.on_trip()
        return cores

    def run(self) -> None:
        self.sample()
        while not self.stop_event.wait(self.period_s):
            self.sample()

    def stop(self) -> None:
        self.stop_event.set()

    def rearm(self) -> None:
        self.tripped = None

    def recent(self, seconds: float) -> list[float]:
        """The samples of the last ``seconds`` (at least two), newest last."""
        count = max(2, int(seconds / self.period_s))
        return [value for _, value in self.samples[-count:]]

    def quiet(self, below_cores: float, seconds: float) -> bool:
        """Foreign work stayed below ``below_cores`` for ``seconds``."""
        count = max(2, int(seconds / self.period_s))
        recent = [value for _, value in self.samples[-count:]]
        return len(recent) == count and max(recent) < below_cores

    def summary(self, start_ns: int, end_ns: int) -> dict[str, object]:
        inside = [value for instant, value in self.samples if start_ns <= instant <= end_ns]
        if not inside:
            return {"samples": 0}
        return {"samples": len(inside), "mean_cores": round(sum(inside) / len(inside), 2),
                "max_cores": max(inside), "samples_above_4": sum(1 for value in inside if value > 4.0)}

    def disturbance(self, start_ns: int, end_ns: int, quiet_cores: float) -> dict[str, object]:
        """How much foreign work above ``quiet_cores`` a window saw (mean excess decides a repeat)."""
        inside = [value for instant, value in self.samples if start_ns <= instant <= end_ns]
        return {"samples": len(inside), "max_cores": max(inside, default=0.0),
                "mean_excess_cores": round(sum(max(0.0, value - quiet_cores) for value in inside)
                                           / max(1, len(inside)), 3)}


def gpu_compute_pids(index: int, *, run: Callable = subprocess.run) -> list[int]:
    """PIDs of the compute processes on one GPU (``nvidia-smi``), ours or not."""
    done = run(["nvidia-smi", "-i", str(index), "--query-compute-apps=pid", "--format=csv,noheader"],
               capture_output=True, text=True, timeout=10, check=True)
    return [int(line) for line in done.stdout.split() if line.strip().isdigit()]


def gpu_memory_used_mib(index: int, *, run: Callable = subprocess.run) -> int:
    done = run(["nvidia-smi", "-i", str(index), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
               capture_output=True, text=True, timeout=10, check=True)
    return int(done.stdout.strip())


def gpu_free(index: int, *, maximum_used_mib: int = 1000, run: Callable = subprocess.run) -> dict[str, object]:
    """State of the serving GPU: free = no compute process and less than ``maximum_used_mib`` in use."""
    pids, used = gpu_compute_pids(index, run=run), gpu_memory_used_mib(index, run=run)
    return {"index": index, "compute_pids": pids, "memory_used_mib": used,
            "free": not pids and used < maximum_used_mib}


def wait_until(check: Callable[[], bool], *, timeout_s: float, period_s: float = 1.0,
               clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
               alive: Callable[[], bool] | None = None) -> float:
    """Seconds waited until ``check()`` held; ``TimeoutError`` after ``timeout_s``; ``alive()`` False ends
    the wait with ``RuntimeError`` (the process waited for is gone)."""
    start = clock()
    while True:
        if check():
            return clock() - start
        if alive is not None and not alive():
            raise RuntimeError("the process being waited for exited")
        if clock() - start > timeout_s:
            raise TimeoutError("condition did not hold within %.0f s" % timeout_s)
        sleep(period_s)
