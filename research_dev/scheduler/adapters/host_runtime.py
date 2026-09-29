"""Host process and telemetry primitives used by physical adapters."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from typing import Callable, Mapping, Sequence

from .contracts import PhysicalAdapterError


@dataclass(frozen=True)
class HostMetricCallbacks:
    gpu_snapshot: Callable[[], Mapping[str, object]]
    rapl_package_snapshot: Callable[[], Mapping[str, object]]
    system_memory: Callable[[], Mapping[str, int]]
    host_activity: Callable[[], Mapping[str, object]] | None = None

    def __post_init__(self) -> None:
        if not all((
            callable(self.gpu_snapshot),
            callable(self.rapl_package_snapshot),
            callable(self.system_memory),
            self.host_activity is None or callable(self.host_activity),
        )):
            raise PhysicalAdapterError("host metric callbacks are invalid")


_CPU_JIFFY_FIELDS = (
    "user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal",
    "guest", "guest_nice",
)
HOST_ACTIVITY_RSS_ALWAYS_BYTES = 256 * 1024 * 1024


def _parse_proc_stat(raw: str) -> tuple[str, list[str]] | None:
    open_index = raw.find("(")
    close_index = raw.rfind(")")
    if open_index < 0 or close_index < open_index:
        return None
    fields = raw[close_index + 2:].split()
    if len(fields) < 22:
        return None
    return raw[open_index + 1:close_index], fields


def linux_host_activity(
    proc_root: Path = Path("/proc"),
    cpu_root: Path = Path("/sys/devices/system/cpu"),
    thread_pids: Sequence[int] | None = None,
) -> dict[str, object]:
    """Raw per-process CPU ticks and RSS, aggregate jiffies and CPU frequencies.

    Counters are cumulative so that deltas can be taken between any two
    samples alongside the RAPL counters; nothing here is a rate. Threads are
    recorded for ``thread_pids`` (default: the parent process, which is the
    runner when this runs inside the forked sampler), with their creation
    time in clock ticks since boot so a thread can be tied to the request
    whose arrival created it even though Linux truncates its name.
    """
    before_ns = time.monotonic_ns()
    first = proc_root.joinpath("stat").read_text(encoding="ascii").splitlines()[0]
    parts = first.split()
    if parts[0] != "cpu" or len(parts) < 8:
        raise PhysicalAdapterError("Linux CPU jiffies are invalid")
    jiffies = {
        name: int(value)
        for name, value in zip(_CPU_JIFFY_FIELDS, parts[1:1 + len(_CPU_JIFFY_FIELDS)])
    }
    page_bytes = os.sysconf("SC_PAGE_SIZE")
    processes = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = entry.joinpath("stat").read_text(encoding="ascii", errors="replace")
        except OSError:
            continue
        parsed = _parse_proc_stat(raw)
        if parsed is None:
            continue
        comm, fields = parsed
        processes.append({
            "pid": int(entry.name),
            "comm": comm,
            "ppid": int(fields[1]),
            "cpu_ticks": int(fields[11]) + int(fields[12]),
            "rss_bytes": int(fields[21]) * page_bytes,
            "start_ticks": int(fields[19]),
        })
    threads = []
    for pid in (thread_pids if thread_pids is not None else (os.getppid(),)):
        task_root = proc_root.joinpath(str(pid), "task")
        try:
            tasks = list(task_root.iterdir())
        except OSError:
            continue
        for task in tasks:
            try:
                raw = task.joinpath("stat").read_text(encoding="ascii", errors="replace")
            except OSError:
                continue
            parsed = _parse_proc_stat(raw)
            if parsed is None:
                continue
            comm, fields = parsed
            threads.append({
                "pid": int(pid),
                "tid": int(task.name),
                "comm": comm,
                "cpu_ticks": int(fields[11]) + int(fields[12]),
                "start_ticks": int(fields[19]),
            })
    frequencies = []
    for cpu in cpu_root.glob("cpu[0-9]*"):
        path = cpu.joinpath("cpufreq", "scaling_cur_freq")
        try:
            frequencies.append(int(path.read_text(encoding="ascii").strip()))
        except (OSError, ValueError):
            continue
    after_ns = time.monotonic_ns()
    return {
        "clock_ticks_per_s": os.sysconf("SC_CLK_TCK"),
        "cpu_jiffies": jiffies,
        "cpu_khz": (
            None if not frequencies else {
                "count": len(frequencies),
                "max": max(frequencies),
                "mean": sum(frequencies) // len(frequencies),
                "min": min(frequencies),
            }
        ),
        "processes": sorted(processes, key=lambda row: row["pid"]),
        "sample_t_ns": (before_ns + after_ns) // 2,
        "scan_ns": after_ns - before_ns,
        "threads": sorted(threads, key=lambda row: (row["pid"], row["tid"])),
    }


def _filter_host_activity(
    activity: Mapping[str, object],
    previous_ticks: dict[int, int],
) -> dict[str, object]:
    """Keep processes that used CPU since the previous sample or hold large RSS."""
    kept = []
    current: dict[int, int] = {}
    for row in activity.get("processes", ()):
        pid = int(row["pid"])
        ticks = int(row["cpu_ticks"])
        current[pid] = ticks
        if (
            previous_ticks.get(pid) != ticks
            or int(row["rss_bytes"]) >= HOST_ACTIVITY_RSS_ALWAYS_BYTES
        ):
            kept.append(dict(row))
    kept_threads = []
    current_threads: dict[tuple[int, int], int] = {}
    for row in activity.get("threads", ()):
        key = (int(row["pid"]), int(row["tid"]))
        ticks = int(row["cpu_ticks"])
        current_threads[key] = ticks
        if previous_ticks.get(key) != ticks:
            kept_threads.append(dict(row))
    previous_ticks.clear()
    previous_ticks.update(current)
    previous_ticks.update(current_threads)
    return {**dict(activity), "processes": kept, "threads": kept_threads}


def linux_system_memory() -> dict[str, int]:
    wanted = {
        "MemAvailable": "available_bytes",
        "SwapFree": "swap_free_bytes",
        "SwapTotal": "swap_total_bytes",
    }
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in wanted:
            fields = value.strip().split()
            if len(fields) != 2 or fields[1] != "kB":
                raise PhysicalAdapterError("Linux memory sample is invalid")
            result[wanted[key]] = int(fields[0]) * 1024
    if set(result) != set(wanted.values()):
        raise PhysicalAdapterError("Linux memory sample is incomplete")
    return result


def nvidia_gpu_snapshot(*, clocks: bool = False) -> dict[str, object]:
    """One ``nvidia-smi`` sample; ``clocks`` adds ``clocks_sm_mhz``, ``clocks_mem_mhz`` and
    ``pstate`` (device power control runs), every other row key is unchanged."""
    if type(clocks) is not bool:
        raise PhysicalAdapterError("NVIDIA GPU sample clock flag is invalid")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.used,memory.free,"
            "utilization.gpu,power.draw"
            + (",clocks.sm,clocks.mem,pstate" if clocks else ""),
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != (10 if clocks else 7):
        raise PhysicalAdapterError("NVIDIA GPU sample is invalid")
    return {
        "memory_free_bytes": int(fields[4]) * 1024 * 1024,
        "memory_total_bytes": int(fields[2]) * 1024 * 1024,
        "memory_used_bytes": int(fields[3]) * 1024 * 1024,
        "name": fields[0],
        "power_mw": int(round(float(fields[6]) * 1000)),
        "utilization_pct": int(fields[5]),
        "uuid": fields[1],
        **({"clocks_mem_mhz": int(fields[8]), "clocks_sm_mhz": int(fields[7]), "pstate": fields[9]}
           if clocks else {}),
    }


def rapl_package_snapshot() -> dict[str, object]:
    root = Path("/sys/class/powercap/intel-rapl:0")
    if root.joinpath("name").read_text(encoding="ascii").strip() != "package-0":
        raise PhysicalAdapterError("RAPL package identity is invalid")
    before_ns = time.monotonic_ns()
    energy_uj = int(root.joinpath("energy_uj").read_text(encoding="ascii"))
    after_ns = time.monotonic_ns()
    maximum_uj = int(
        root.joinpath("max_energy_range_uj").read_text(encoding="ascii")
    )
    if maximum_uj <= 0 or not 0 <= energy_uj < maximum_uj:
        raise PhysicalAdapterError("RAPL package sample is invalid")
    return {
        "energy_uj": energy_uj,
        "max_energy_range_uj": maximum_uj,
        "name": "package-0",
        "sample_t_ns": (before_ns + after_ns) // 2,
    }


def default_host_metric_callbacks(*, gpu_clocks: bool = False) -> HostMetricCallbacks:
    """The host samplers; ``gpu_clocks`` (device power control) adds the GPU clock columns."""
    if type(gpu_clocks) is not bool:
        raise PhysicalAdapterError("host metric GPU clock flag is invalid")
    return HostMetricCallbacks(
        gpu_snapshot=(lambda: nvidia_gpu_snapshot(clocks=True)) if gpu_clocks else nvidia_gpu_snapshot,
        rapl_package_snapshot=rapl_package_snapshot,
        system_memory=linux_system_memory,
        host_activity=linux_host_activity,
    )


def _run_host_energy_sampler(
    callbacks: HostMetricCallbacks,
    interval_s: float,
    stop_event: object,
    messages: object,
) -> None:
    previous_ticks: dict[int, int] = {}
    try:
        while not stop_event.is_set():
            try:
                before_ns = time.monotonic_ns()
                gpu = dict(callbacks.gpu_snapshot())
                after_ns = time.monotonic_ns()
            except Exception as error:
                messages.put({
                    "kind": "probe_error",
                    "event": {
                        "event": "HOST_METRIC_PROBE_ERROR",
                        "host_sample_t_ns": time.monotonic_ns(),
                        "reason": f"{type(error).__name__}: {error}",
                        "source": "gpu",
                    },
                })
                stop_event.wait(interval_s)
                continue
            gpu["sample_t_ns"] = (before_ns + after_ns) // 2
            try:
                rapl = dict(callbacks.rapl_package_snapshot())
            except Exception as error:
                messages.put({
                    "kind": "probe_error",
                    "event": {
                        "event": "HOST_METRIC_PROBE_ERROR",
                        "host_sample_t_ns": time.monotonic_ns(),
                        "reason": f"{type(error).__name__}: {error}",
                        "source": "rapl_package",
                    },
                })
                rapl = None
            try:
                system = dict(callbacks.system_memory())
            except Exception as error:
                messages.put({
                    "kind": "probe_error",
                    "event": {
                        "event": "HOST_METRIC_PROBE_ERROR",
                        "host_sample_t_ns": time.monotonic_ns(),
                        "reason": f"{type(error).__name__}: {error}",
                        "source": "system_memory",
                    },
                })
                stop_event.wait(interval_s)
                continue
            row = {
                "gpu": gpu,
                "rapl_package": rapl,
                "schema": "heterogeneous-host-sample-v1",
                "system": system,
                "t_ns": time.monotonic_ns(),
            }
            if callbacks.host_activity is not None:
                try:
                    row["host_activity"] = _filter_host_activity(
                        callbacks.host_activity(), previous_ticks
                    )
                except Exception as error:
                    messages.put({
                        "kind": "probe_error",
                        "event": {
                            "event": "HOST_METRIC_PROBE_ERROR",
                            "host_sample_t_ns": time.monotonic_ns(),
                            "reason": f"{type(error).__name__}: {error}",
                            "source": "host_activity",
                        },
                    })
            messages.put({
                "clear_probe_error": rapl is not None,
                "kind": "row",
                "row": row,
            })
            stop_event.wait(interval_s)
    except BaseException as error:
        try:
            messages.put({
                "kind": "fatal",
                "reason": f"{type(error).__name__}: {error}",
            })
        except BaseException:
            pass


class CapturedProcess:
    """Run a process group while preserving line-buffered diagnostic output."""

    def __init__(
        self,
        command: Sequence[str],
        environment: Mapping[str, str],
        output_directory: Path,
        label: str,
    ) -> None:
        if (
            not command
            or any(type(value) is not str or not value for value in command)
            or not isinstance(output_directory, Path)
            or type(label) is not str
            or not label
            or not label.isascii()
        ):
            raise PhysicalAdapterError("captured process setup is invalid")
        self.command = tuple(command)
        self.environment = dict(environment)
        self.output_directory = output_directory
        self.label = label
        self.process: subprocess.Popen[str] | None = None
        self.stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._stderr_condition = threading.Condition()
        self._stderr_file = None
        self._stdout_file = None

    def start(self) -> None:
        if self.process is not None:
            raise PhysicalAdapterError("captured process already started")
        self._stderr_file = (
            self.output_directory / (self.label + ".stderr")
        ).open("x", encoding="utf-8")
        self._stdout_file = (
            self.output_directory / (self.label + ".stdout")
        ).open("x", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=self._stdout_file,
            stderr=subprocess.PIPE,
            env=self.environment,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            bufsize=1,
            start_new_session=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name=self.label + "-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _read_stderr(self) -> None:
        process = self.process
        stream = self._stderr_file
        if process is None or process.stderr is None or stream is None:
            return
        for line in process.stderr:
            stream.write(line)
            stream.flush()
            with self._stderr_condition:
                self.stderr_lines.append(line.rstrip("\n"))
                self._stderr_condition.notify_all()

    def wait_stderr(self, prefix: str, timeout_s: float) -> str:
        if type(prefix) is not str or not prefix or timeout_s <= 0:
            raise PhysicalAdapterError("captured process wait is invalid")
        deadline = time.monotonic() + timeout_s
        with self._stderr_condition:
            while True:
                for line in self.stderr_lines:
                    if line.startswith(prefix):
                        return line
                if self.process is None or self.process.poll() is not None:
                    raise PhysicalAdapterError(
                        self.label + ": exited before " + prefix
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PhysicalAdapterError(
                        self.label + ": timeout waiting for " + prefix
                    )
                self._stderr_condition.wait(min(remaining, 0.2))

    def terminate(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        for name in ("_stderr_file", "_stdout_file"):
            stream = getattr(self, name)
            if stream is not None:
                stream.close()
                setattr(self, name, None)


class HostEnergySampler:
    """Collect raw host observations without campaign persistence policy."""

    def __init__(
        self,
        callbacks: HostMetricCallbacks,
        *,
        interval_s: float = 0.2,
    ) -> None:
        if not isinstance(callbacks, HostMetricCallbacks) or interval_s <= 0:
            raise PhysicalAdapterError("host energy sampler setup is invalid")
        self._callbacks = callbacks
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._sync_lock = threading.Lock()
        try:
            context = multiprocessing.get_context("fork")
        except ValueError as error:
            raise PhysicalAdapterError(
                "host energy sampler requires process isolation"
            ) from error
        self._stop = context.Event()
        self._messages = context.Queue()
        self._process = context.Process(
            target=_run_host_energy_sampler,
            args=(
                self._callbacks,
                self._interval_s,
                self._stop,
                self._messages,
            ),
            name="heterogeneous-host-energy",
            daemon=True,
        )
        self._started = False
        self._error: str | None = None
        self._last_probe_error: str | None = None
        self._events: list[dict[str, object]] = []
        self._rows: list[dict[str, object]] = []

    def start(self) -> None:
        if self._started:
            raise PhysicalAdapterError("host energy sampler already started")
        self._process.start()
        self._started = True

    def _sync(self) -> None:
        with self._sync_lock:
            received = []
            while True:
                try:
                    received.append(self._messages.get_nowait())
                except queue.Empty:
                    break
            if not received:
                return
            with self._lock:
                for message in received:
                    kind = message.get("kind")
                    if kind == "row":
                        self._rows.append(dict(message["row"]))
                        if message.get("clear_probe_error") is True:
                            self._last_probe_error = None
                    elif kind == "probe_error":
                        event = dict(message["event"])
                        self._events.append(event)
                        self._last_probe_error = str(event["reason"])
                    elif kind == "fatal":
                        self._error = str(message.get("reason"))
                    else:
                        self._error = "host sampler message is invalid"

    def rows(self) -> tuple[dict[str, object], ...]:
        self._sync()
        with self._lock:
            return tuple(dict(row) for row in self._rows)

    def latest_rows(self) -> tuple[dict[str, object], ...]:
        self._sync()
        with self._lock:
            return tuple(dict(row) for row in self._rows[-4:])

    def rows_between(
        self, start_ns: int, end_ns: int
    ) -> tuple[dict[str, object], ...]:
        if type(start_ns) is not int or type(end_ns) is not int:
            raise PhysicalAdapterError("host energy interval is invalid")
        self._sync()
        with self._lock:
            rows = tuple(
                row for row in self._rows
                if type(row.get("gpu")) is dict
                and type(row.get("rapl_package")) is dict
            )
            if not rows:
                return ()
            before = [
                index for index, row in enumerate(rows)
                if int(row["gpu"]["sample_t_ns"]) <= start_ns
                and int(row["rapl_package"]["sample_t_ns"]) <= start_ns
            ]
            after = [
                index for index, row in enumerate(rows)
                if int(row["gpu"]["sample_t_ns"]) >= end_ns
                and int(row["rapl_package"]["sample_t_ns"]) >= end_ns
            ]
            first = before[-1] if before else 0
            last = after[0] if after else len(rows) - 1
            return tuple(dict(row) for row in rows[first:last + 1])

    def latest_gpu(self) -> dict[str, object]:
        self._sync()
        with self._lock:
            if not self._rows:
                raise PhysicalAdapterError("GPU observation is absent")
            return dict(self._rows[-1]["gpu"])

    def diagnostics(self) -> dict[str, object]:
        self._sync()
        with self._lock:
            return {
                "events": [dict(row) for row in self._events],
                "fatal_error": self._error,
                "last_probe_error": self._last_probe_error,
                "process_exitcode": self._process.exitcode,
                "sample_count": len(self._rows),
                "schema": "heterogeneous-host-power-diagnostics-v1",
            }

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        deadline = time.monotonic() + 15
        while self._process.is_alive() and time.monotonic() < deadline:
            self._sync()
            self._process.join(timeout=0.05)
        self._sync()
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        if (
            self._process.is_alive()
            or self._process.exitcode != 0
            or self._error is not None
        ):
            raise PhysicalAdapterError("host energy sampler failed")


def _interpolate(
    points: Sequence[tuple[int, float]], target_ns: int
) -> float:
    if len(points) < 2 or not points[0][0] <= target_ns <= points[-1][0]:
        raise PhysicalAdapterError("energy sample coverage is incomplete")
    for left, right in zip(points, points[1:]):
        if target_ns <= right[0]:
            if right[0] <= left[0]:
                raise PhysicalAdapterError("energy sample order is invalid")
            fraction = (target_ns - left[0]) / (right[0] - left[0])
            return left[1] + fraction * (right[1] - left[1])
    raise PhysicalAdapterError("energy interpolation coverage is incomplete")


def _integrate_gpu(
    rows: Sequence[Mapping[str, object]], start_ns: int, end_ns: int
) -> float:
    points = sorted(
        (
            int(row["gpu"]["sample_t_ns"]),
            float(row["gpu"]["power_mw"]) / 1000,
        )
        for row in rows
    )
    bounded = [(start_ns, _interpolate(points, start_ns))]
    bounded.extend(point for point in points if start_ns < point[0] < end_ns)
    bounded.append((end_ns, _interpolate(points, end_ns)))
    return sum(
        (right[0] - left[0]) / 1e9 * (left[1] + right[1]) / 2
        for left, right in zip(bounded, bounded[1:])
    )


def _integrate_rapl(
    rows: Sequence[Mapping[str, object]], start_ns: int, end_ns: int
) -> float:
    samples = sorted(
        (
            int(row["rapl_package"]["sample_t_ns"]),
            int(row["rapl_package"]["energy_uj"]),
            int(row["rapl_package"]["max_energy_range_uj"]),
        )
        for row in rows
    )
    if len(samples) != len(rows) or len(samples) < 2:
        raise PhysicalAdapterError("RAPL sample coverage is incomplete")
    maximum = samples[0][2]
    if maximum <= 0 or any(row[2] != maximum for row in samples):
        raise PhysicalAdapterError("RAPL range changed")
    unwrapped: list[tuple[int, float]] = [(samples[0][0], 0.0)]
    prior = samples[0][1]
    total = 0
    for sample_ns, energy_uj, _ in samples[1:]:
        delta = energy_uj - prior
        if delta < 0:
            delta += maximum
        if not 0 <= delta < maximum:
            raise PhysicalAdapterError("RAPL counter delta is invalid")
        total += delta
        unwrapped.append((sample_ns, float(total)))
        prior = energy_uj
    return (
        _interpolate(unwrapped, end_ns)
        - _interpolate(unwrapped, start_ns)
    ) / 1e6


def server_energy_summary(
    rows: Sequence[Mapping[str, object]], start_ns: int, end_ns: int
) -> Mapping[str, float | str]:
    if type(start_ns) is not int or type(end_ns) is not int or end_ns <= start_ns:
        raise PhysicalAdapterError("server energy interval is invalid")
    gpu_energy_j = _integrate_gpu(rows, start_ns, end_ns)
    cpu_energy_j = _integrate_rapl(rows, start_ns, end_ns)
    duration_s = (end_ns - start_ns) / 1e9
    return {
        "boundary": "paid_trace_interval",
        "cpu_package_average_power_w": cpu_energy_j / duration_s,
        "cpu_package_energy_j": cpu_energy_j,
        "gpu_board_average_power_w": gpu_energy_j / duration_s,
        "gpu_board_energy_j": gpu_energy_j,
        "method": "RAPL package delta plus trapezoidal GPU board power",
        "server_compute_device_energy_j": cpu_energy_j + gpu_energy_j,
    }
