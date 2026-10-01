"""Opt-in, scheduler-driven power control of the desktop GPU (and CPU EPP).

Measured on the RTX 4060 Ti desktop during a two-phone trace: the GPU is 8.8 % utilized yet draws
34 W while a model decodes, 27.6 W with weights resident but idle and 11 W unloaded (P8: 210 MHz
SM); it is 49 % of host energy. The controller lowers the SM clock (and optionally the intel_pstate
energy-performance preference) while the scheduler knows the GPU has nothing to do, and restores
them before the next known GPU work. It only knows what the rig and the runner tell it: active GPU
executions, the model-load window of a transition, the next trace arrival and the earliest queued
ticket. Without a ``device_power`` policy there is no controller and behaviour is byte-identical.

``arrival_information: "online"`` replaces the arrival calendar with arrivals that already happened
(``note_arrival_observed``; ``note_next_arrival_us`` is refused) and an ``ArrivalGapPredictor``; an
explicit ``arrival_information`` also writes ``DEVICE_POWER_TELEMETRY.json`` (idle intervals, the
synchronous wait of every GPU execution start, observed arrivals). ``decode_cap.protect_prefill``
keeps full clocks while any active execution has not produced its first token.

Fail-closed: every command is ``sudo -n`` on the exact argv the desktop sudoers rule allows; the
first non-zero exit, timeout or OSError flips the controller to UNAVAILABLE and no further command
is issued (behaviour = policy off), except one best-effort restore at ``close`` when a lock ever
succeeded so the GPU is never left locked after the run. No command ever runs under the rig or the
scheduler lock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import copy
import json
import math
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable, Protocol, Sequence

from ..configuration.campaign import CPU_EPP_VALUES, DevicePowerConfiguration
from .contracts import PhysicalAdapterError
from .ticket import PhysicalExecutionCommand

# The campaign ``device_power`` object is the policy: one validator, one frozen type.
DevicePowerPolicy = DevicePowerConfiguration

NVIDIA_SMI = "/usr/bin/nvidia-smi"
TEE = "/usr/bin/tee"
SUDO = ("sudo", "-n")
CPU_ROOT = Path("/sys/devices/system/cpu")
CPU_EPP_FILE = "cpufreq/energy_performance_preference"
COMMAND_TIMEOUT_S = 5.0
DEVICE_POWER_STATES = ("OFF", "UNAVAILABLE", "RESTORED", "IDLE_MIN", "DECODE_CAP", "LOAD_MIN")
DEVICE_POWER_EVENT_KIND = "DEVICE_POWER_STATE"
DEVICE_POWER_EVENTS_FILE = "DEVICE_POWER_EVENTS.json"
DEVICE_POWER_TELEMETRY_FILE = "DEVICE_POWER_TELEMETRY.json"
DEVICE_POWER_TELEMETRY_SCHEMA = "device-power-telemetry-v1"
# online idle rule: a predicted drop waits this long after the GPU went idle, which covers the
# finish -> next dispatch and transition -> execution hand-offs (s2a: at most ~0.31 s)
ONLINE_IDLE_SETTLE_US = 1_000_000
# states in which the controller issues no command
_INERT_STATES = ("OFF", "UNAVAILABLE")

CommandResult = dict[str, object]


def run_control_command(
    run: Callable[..., subprocess.CompletedProcess],
    argv: Sequence[str],
    *,
    input_text: str | None = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> CommandResult:
    """Run one control command and return its result row; never raises on the environment.

    Row: ``command`` (argv), ``returncode`` (None on timeout/OSError), ``stderr`` (the failure
    text on timeout/OSError), ``readback`` (None; the caller fills it), ``duration_us``. The
    command succeeded iff ``returncode == 0``."""
    started_ns = monotonic_ns()
    options = {} if input_text is None else {"input": input_text}
    try:
        completed = run(list(argv), capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S,
                        check=False, **options)
    except subprocess.TimeoutExpired as error:
        returncode, stderr, stdout = None, "TimeoutExpired: " + str(error), ""
    except OSError as error:
        returncode, stderr, stdout = None, type(error).__name__ + ": " + str(error), ""
    else:
        returncode = completed.returncode
        stderr = "" if completed.stderr is None else str(completed.stderr)
        stdout = "" if completed.stdout is None else str(completed.stdout)
    return {
        "command": list(argv),
        "returncode": returncode,
        "stderr": stderr.strip(),
        "stdout": stdout.strip(),
        "readback": None,
        "duration_us": max(0, (monotonic_ns() - started_ns) // 1000),
    }


def command_succeeded(result: CommandResult) -> bool:
    return result.get("returncode") == 0


class DeviceClockControl(Protocol):
    """Clock control of one device. ``lock`` pins the compute clock to ``[min_mhz, max_mhz]``,
    ``restore`` lifts every lock this control can set, ``probe`` checks that the exact argv of
    those commands is permitted without running them, ``readback`` reports the current clocks.
    Every method returns a result row (``run_control_command``) and never raises on the
    environment. A phone HTP DCVS control implements the same four methods."""

    def probe(self, lock_pairs: Sequence[tuple[int, int]]) -> tuple[CommandResult, ...]: ...

    def lock(self, min_mhz: int, max_mhz: int) -> CommandResult: ...

    def restore(self) -> CommandResult: ...

    def readback(self) -> dict[str, object]: ...


class NvidiaClockControl:
    """``nvidia-smi`` SM clock lock of one board. ``-i <uuid>`` follows ``-lgc <min>,<max>`` so the
    argv matches the sudoers rule ``nvidia-smi -lgc *``; ``-rgc`` is bare because its rule carries
    no wildcard (it resets every board's SM lock; the desktop has one). Memory clocks are never
    touched. The readback (``clocks.sm,clocks.mem,pstate``) runs without sudo."""

    def __init__(
        self,
        gpu_uuid: str,
        *,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if type(gpu_uuid) is not str or not gpu_uuid or not gpu_uuid.isascii():
            raise PhysicalAdapterError("NVIDIA clock control GPU uuid is invalid")
        self.gpu_uuid = gpu_uuid
        self._run = run
        self._monotonic_ns = monotonic_ns

    def _lock_argv(self, min_mhz: int, max_mhz: int) -> tuple[str, ...]:
        if type(min_mhz) is not int or type(max_mhz) is not int or not 0 < min_mhz <= max_mhz:
            raise PhysicalAdapterError("NVIDIA clock lock range is invalid")
        return (NVIDIA_SMI, "-lgc", f"{min_mhz},{max_mhz}", "-i", self.gpu_uuid)

    @staticmethod
    def _restore_argv() -> tuple[str, ...]:
        return (NVIDIA_SMI, "-rgc")

    def _command(self, argv: Sequence[str]) -> CommandResult:
        result = run_control_command(self._run, (*SUDO, *argv), monotonic_ns=self._monotonic_ns)
        if command_succeeded(result):
            result["readback"] = self.readback()
        return result

    def probe(self, lock_pairs: Sequence[tuple[int, int]]) -> tuple[CommandResult, ...]:
        """``sudo -n -l <argv>`` for every lock pair and for the restore: exit 0 iff permitted."""
        argvs = [self._lock_argv(*pair) for pair in lock_pairs] + [self._restore_argv()]
        return tuple(
            run_control_command(self._run, (*SUDO, "-l", *argv), monotonic_ns=self._monotonic_ns)
            for argv in argvs
        )

    def lock(self, min_mhz: int, max_mhz: int) -> CommandResult:
        return self._command(self._lock_argv(min_mhz, max_mhz))

    def restore(self) -> CommandResult:
        return self._command(self._restore_argv())

    def readback(self) -> dict[str, object]:
        """Current SM/memory clocks and pstate, or ``{"error": ...}``; diagnostic only."""
        result = run_control_command(
            self._run,
            (NVIDIA_SMI, "-i", self.gpu_uuid, "--query-gpu=clocks.sm,clocks.mem,pstate",
             "--format=csv,noheader,nounits"),
            monotonic_ns=self._monotonic_ns,
        )
        if not command_succeeded(result):
            return {"error": "nvidia-smi query exit " + str(result["returncode"]) + ": " + str(result["stderr"])}
        fields = [field.strip() for field in str(result["stdout"]).split(",")]
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            return {"error": "nvidia-smi clock readback is invalid: " + str(result["stdout"])}
        return {"clocks_sm_mhz": int(fields[0]), "clocks_mem_mhz": int(fields[1]), "pstate": fields[2]}


class CpuEppControl:
    """intel_pstate energy_performance_preference of every CPU, written with one ``sudo -n tee``
    over the expanded ``cpu*/cpufreq/energy_performance_preference`` paths (the sudoers wildcard
    matches the whole argument list; ``probe`` verifies that exact argv with ``sudo -l``). The
    value read from cpu0 before the first write is what ``restore`` puts back."""

    def __init__(
        self,
        *,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        cpu_root: Path = CPU_ROOT,
    ) -> None:
        self._run = run
        self._monotonic_ns = monotonic_ns
        self._cpu_root = cpu_root

    def targets(self) -> tuple[str, ...]:
        """The EPP files of every CPU, cpu0 first; empty when the platform has none."""
        found = []
        for path in self._cpu_root.glob("cpu[0-9]*/" + CPU_EPP_FILE):
            found.append((int(path.parts[-3][3:]), str(path)))
        return tuple(path for _, path in sorted(found))

    def read(self) -> str | None:
        """The EPP of cpu0 (one ASCII token), None when unreadable or when no CPU exposes it."""
        targets = self.targets()
        if not targets:
            return None
        try:
            value = Path(targets[0]).read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            return None
        return value if value and value.isascii() and " " not in value else None

    def probe(self) -> tuple[CommandResult, ...]:
        targets = self.targets()
        if not targets:
            return ({"command": [TEE], "returncode": None, "stderr": "no cpufreq EPP files under " +
                     str(self._cpu_root), "stdout": "", "readback": None, "duration_us": 0},)
        return (run_control_command(self._run, (*SUDO, "-l", TEE, *targets), monotonic_ns=self._monotonic_ns),)

    def write(self, value: str) -> CommandResult:
        if value not in CPU_EPP_VALUES:
            raise PhysicalAdapterError("CPU EPP value is invalid")
        targets = self.targets()
        result = run_control_command(self._run, (*SUDO, TEE, *targets), input_text=value + "\n",
                                     monotonic_ns=self._monotonic_ns)
        if command_succeeded(result):
            result["readback"] = self.read()
            if result["readback"] != value:
                result["returncode"] = None if result["returncode"] is None else -1
                result["stderr"] = "EPP readback differs from the written value"
        return result


@dataclass
class _ControllerInputs:
    """Everything the idle rule reads; owned by the controller's condition."""

    active_gpu: dict[str, str]  # ticket_id -> executor_id of active GPU-targeting executions
    load_active: bool = False
    transition_active: bool = False
    next_arrival_us: int | None = None
    arrivals_finished: bool = False
    queued_start_us: int | None = None
    frozen: bool = False  # end_trace: no new lock, restore only
    # decode_cap.protect_prefill: ticket -> request of active executions, and those past first token
    request_by_ticket: dict[str, str] = field(default_factory=dict)
    decoding: set[str] = field(default_factory=set)
    # online: observation time of the last arrival whose GPU work has not started yet
    pending_arrival_us: int | None = None


class ArrivalGapPredictor:
    """Pessimistic online estimate of "an arrival within ``horizon_us``" from observed arrivals.

    Every observed arrival closes one inter-arrival gap (one sequence, or one per model). With
    ``e`` the time since the last arrival, ``n`` the observed gaps longer than ``e`` and ``k`` of
    them ending within ``e + horizon``, a sequence estimates ``(k + 1) / (n + 1)``: 1 without
    evidence, and it can only fall to ``q`` after ``1/q - 1`` surviving gaps. Per-model sequences
    combine as ``1 - prod(1 - p_m)``. Only timestamps passed to ``observe`` ever enter."""

    def __init__(self, horizon_us: int, *, per_model: bool) -> None:
        if type(horizon_us) is not int or horizon_us <= 0 or type(per_model) is not bool:
            raise PhysicalAdapterError("arrival predictor setup is invalid")
        self._horizon_us = horizon_us
        self._per_model = per_model
        self._last: dict[str, int] = {}
        self._gaps: dict[str, list[int]] = {}

    def observe(self, model_id: str, observed_at_us: int) -> None:
        key = model_id if self._per_model else "*"
        previous = self._last.get(key)
        if previous is not None:
            if observed_at_us < previous:
                raise PhysicalAdapterError("arrival predictor observation is out of order")
            self._gaps.setdefault(key, []).append(observed_at_us - previous)
        self._last[key] = observed_at_us

    def estimate(self, now_us: int) -> dict[str, object] | None:
        """``{probability_ppm (ceil), sequences: [{key, elapsed_us, gaps, at_risk, within}]}``,
        None before the first observed arrival."""
        if not self._last:
            return None
        survive = Fraction(1)
        rows = []
        for key in sorted(self._last):
            elapsed = max(0, now_us - self._last[key])
            gaps = self._gaps.get(key, ())
            at_risk = [gap for gap in gaps if gap > elapsed]
            within = sum(1 for gap in at_risk if gap <= elapsed + self._horizon_us)
            survive *= Fraction(len(at_risk) - within, len(at_risk) + 1)
            rows.append({"key": key, "elapsed_us": elapsed, "gaps": len(gaps), "at_risk": len(at_risk),
                         "within": within})
        return {"probability_ppm": math.ceil((1 - survive) * 1_000_000), "sequences": rows}


def device_power_capability(
    policy: DevicePowerPolicy,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    cpu_root: Path = CPU_ROOT,
) -> tuple[bool, str, tuple[CommandResult, ...]]:
    """``sudo -n -l`` the exact commands the policy needs (and read the current EPP when the policy
    sets one). Returns (available, one-line ASCII detail, probe rows); never raises."""
    if not isinstance(policy, DevicePowerPolicy):
        raise PhysicalAdapterError("device power policy is invalid")
    gpu = NvidiaClockControl(policy.gpu_uuid, run=run)
    rows = list(gpu.probe(_lock_pairs(policy)))
    detail = []
    if policy.idle is not None and policy.idle.cpu_epp is not None:
        epp = CpuEppControl(run=run, cpu_root=cpu_root)
        rows.extend(epp.probe())
        initial = epp.read()
        if initial is None:
            rows.append({"command": ["read", CPU_EPP_FILE], "returncode": None,
                         "stderr": "CPU EPP is unreadable", "stdout": "", "readback": None, "duration_us": 0})
        else:
            detail.append("EPP " + initial)
    failed = [row for row in rows if not command_succeeded(row)]
    if failed:
        first = failed[0]
        detail.insert(0, "refused: " + " ".join(str(part) for part in first["command"]) + " -> "
                      + ("exit " + str(first["returncode"]) if first["returncode"] is not None else "no exit")
                      + (" " + str(first["stderr"]) if first["stderr"] else ""))
    else:
        detail.insert(0, "sudo permits nvidia-smi -lgc/-rgc for " + policy.gpu_uuid
                      + (" and tee EPP" if policy.idle is not None and policy.idle.cpu_epp is not None else ""))
    text = "; ".join(detail).encode("ascii", "replace").decode("ascii")
    return not failed, text, tuple(rows)


def _lock_pairs(policy: DevicePowerPolicy) -> tuple[tuple[int, int], ...]:
    """Every distinct ``-lgc`` pair the policy may issue (idle/load lock, decode cap range)."""
    if policy.idle is None:
        return ()
    floor = policy.idle.gpu_min_clocks_mhz
    pairs = [(floor, floor)]
    if policy.decode_cap is not None:
        pairs.append((floor, policy.decode_cap.sm_max_mhz))
    return tuple(pairs)


class DevicePowerController:
    """State machine over OFF, UNAVAILABLE, RESTORED, IDLE_MIN, DECODE_CAP and LOAD_MIN.

    Inputs are non-blocking setters (``note_*``, ``on_execution_finish``, ``on_server_stopped``)
    plus the synchronous hooks that must take effect before GPU work starts (``on_execution_start``
    late restore, ``on_load_begin``/``on_load_end``). A daemon thread converges the hardware to the
    target state on every input change and at ``tick_interval_s``; the target is:

    * a GPU-targeting execution is active -> DECODE_CAP when ``decode_cap`` is set, else RESTORED;
    * a model load is in progress -> LOAD_MIN when ``load_min`` is set, else RESTORED;
    * a transition is active, a ticket is queued, the trace ended or the next arrival is unknown
      -> RESTORED;
    * otherwise the gap to the next arrival decides: in IDLE_MIN stay until it is shorter than
      ``lead_ms`` (predictive restore); in RESTORED enter IDLE_MIN once it reaches ``min_gap_s``.

    Online (``arrival_information: "online"``) the last rule is: RESTORED before the first observed
    arrival and for ``min_gap_s`` after an observed arrival whose GPU work has not started; IDLE_MIN
    stays until one of the rules above restores; RESTORED enters IDLE_MIN when the predictor's
    estimate of an arrival within ``min_gap_s`` is at most ``max_arrival_probability_ppm`` (after
    ``ONLINE_IDLE_SETTLE_US`` of idleness) or the GPU has been idle for ``fallback_idle_ms``. With
    ``protect_prefill`` an active execution before its first token keeps RESTORED.

    Every state change is one ``DEVICE_POWER_STATE`` event (``at_us`` on the RESULT clock, the
    commands issued and their result rows), persisted to ``DEVICE_POWER_EVENTS.json``."""

    def __init__(
        self,
        policy: DevicePowerPolicy,
        *,
        epoch_ns_provider: Callable[[], int],
        output_directory: Path,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        clock: DeviceClockControl | None = None,
        epp: CpuEppControl | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        tick_interval_s: float = 0.5,
        queued_start_provider: Callable[[], int | None] | None = None,
    ) -> None:
        if not isinstance(policy, DevicePowerPolicy):
            raise PhysicalAdapterError("device power policy is invalid")
        if not callable(epoch_ns_provider) or not isinstance(output_directory, Path):
            raise PhysicalAdapterError("device power controller setup is invalid")
        if type(tick_interval_s) not in (int, float) or tick_interval_s <= 0:
            raise PhysicalAdapterError("device power tick interval is invalid")
        self.policy = policy
        self._epoch_ns_provider = epoch_ns_provider
        self._output_directory = output_directory
        self._monotonic_ns = monotonic_ns
        self._tick_interval_s = float(tick_interval_s)
        self._clock = clock if clock is not None else NvidiaClockControl(policy.gpu_uuid, run=run, monotonic_ns=monotonic_ns)
        wants_epp = policy.idle is not None and policy.idle.cpu_epp is not None
        self._epp = epp if epp is not None else (CpuEppControl(run=run, monotonic_ns=monotonic_ns) if wants_epp else None)
        self._queued_start_provider = queued_start_provider
        self._condition = threading.Condition()
        self._command_lock = threading.Lock()
        self._telemetry_file_lock = threading.Lock()
        self._inputs = _ControllerInputs(active_gpu={})
        self._state = "OFF"
        self._events: list[dict[str, object]] = []
        self._gpu_lock: tuple[int, int] | None = None
        self._epp_initial: str | None = None
        self._epp_applied: str | None = None
        self._ever_locked = False
        self._wake_timeout_s = self._tick_interval_s
        self._stop = False
        self._thread: threading.Thread | None = None
        self._protect_prefill = policy.decode_cap is not None and policy.decode_cap.protect_prefill
        self._telemetry = policy.arrival_information is not None
        self._predictor = (
            ArrivalGapPredictor(policy.idle.min_gap_s * 1_000_000, per_model=policy.online_idle.predictor == "per_model")
            if policy.online else None)
        self._decision: dict[str, object] | None = None  # the online rule's inputs of the last target
        self._idle_since_ns: int | None = None
        self._idle_intervals: list[dict[str, object]] = []
        self._execution_waits: list[dict[str, object]] = []
        self._observed_arrivals: list[dict[str, object]] = []

    # ---- observation ------------------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    @property
    def events(self) -> tuple[dict[str, object], ...]:
        with self._condition:
            return tuple(copy.deepcopy(row) for row in self._events)

    def _at_us(self) -> int:
        return max(0, (self._monotonic_ns() - self._epoch_ns_provider()) // 1000)

    def _record(self, previous: str, state: str, reason: str, results: Sequence[CommandResult],
                decision: dict[str, object] | None = None) -> None:
        row = {
            "kind": DEVICE_POWER_EVENT_KIND,
            "device": self.policy.device,
            "from": previous,
            "to": state,
            "reason": reason,
            "at_us": self._at_us(),
            "command": [list(result["command"]) for result in results],
            "result": [copy.deepcopy(result) for result in results],
        }
        if decision is not None:
            row["decision"] = copy.deepcopy(decision)
        with self._condition:
            self._state = state
            self._events.append(row)
            events = copy.deepcopy(self._events)
            self._condition.notify_all()
        (self._output_directory / DEVICE_POWER_EVENTS_FILE).write_text(
            json.dumps(events, indent=2, sort_keys=True) + "\n", encoding="ascii")

    # ---- telemetry (explicit arrival_information only) ----------------------------------------

    @property
    def telemetry(self) -> dict[str, object] | None:
        """Idle intervals, execution-start waits and observed arrivals; None without
        ``arrival_information``. An open idle interval is reported up to now."""
        if not self._telemetry:
            return None
        with self._condition:
            return self._telemetry_locked()

    def _telemetry_locked(self) -> dict[str, object]:
        intervals = copy.deepcopy(self._idle_intervals)
        if self._idle_since_ns is not None:
            intervals.append({"start_us": self._ns_to_us(self._idle_since_ns), "end_us": self._at_us(),
                              "end_reason": "open"})
        online = self.policy.online_idle
        return {
            "schema": DEVICE_POWER_TELEMETRY_SCHEMA,
            "device": self.policy.device,
            "arrival_information": self.policy.arrival_information,
            "horizon_us": None if self.policy.idle is None else self.policy.idle.min_gap_s * 1_000_000,
            "online_idle": None if online is None else online.to_json(),
            "protect_prefill": self._protect_prefill,
            "idle_intervals": intervals,
            "execution_waits": copy.deepcopy(self._execution_waits),
            "observed_arrivals": copy.deepcopy(self._observed_arrivals),
        }

    def _ns_to_us(self, value_ns: int) -> int:
        return max(0, (value_ns - self._epoch_ns_provider()) // 1000)

    def _idle_blocker_locked(self) -> str | None:
        """Why the GPU is not idle for the power rule (None = idle); arrivals never count."""
        inputs = self._inputs
        if inputs.active_gpu:
            return "execution_active"
        if inputs.load_active:
            return "load"
        if inputs.frozen:
            return "trace_ended"
        if inputs.transition_active:
            return "transition_active"
        if inputs.queued_start_us is not None:
            return "ticket_queued"
        return None

    def _track_idle_locked(self) -> bool:
        """Open or close the current idle interval after an input change; True when one closed."""
        if not self._telemetry or self._state in _INERT_STATES:
            return False
        blocker = self._idle_blocker_locked()
        if blocker is None and self._idle_since_ns is None:
            self._idle_since_ns = self._monotonic_ns()
        elif blocker is not None and self._idle_since_ns is not None:
            self._idle_intervals.append({"start_us": self._ns_to_us(self._idle_since_ns), "end_us": self._at_us(),
                                         "end_reason": blocker})
            self._idle_since_ns = None
            return True
        return False

    def _persist_telemetry(self) -> None:
        if not self._telemetry:
            return
        with self._telemetry_file_lock:
            with self._condition:
                row = self._telemetry_locked()
            (self._output_directory / DEVICE_POWER_TELEMETRY_FILE).write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="ascii")

    # ---- lifecycle --------------------------------------------------------------------------

    def capability_probe(self) -> bool:
        """``sudo -n -l`` every command the policy needs, read the initial EPP and lift any stale
        lock (``-rgc``) so the run starts from a known state: OFF -> RESTORED, or -> UNAVAILABLE
        with one event carrying the refused row. Returns whether the controller is available."""
        with self._condition:
            if self._state != "OFF":
                raise PhysicalAdapterError("device power controller is already probed")
        with self._command_lock:
            rows = list(self._clock.probe(_lock_pairs(self.policy)))
            if self._epp is not None:
                rows.extend(self._epp.probe())
                initial = self._epp.read()
                if initial is None:
                    rows.append({"command": ["read", CPU_EPP_FILE], "returncode": None,
                                 "stderr": "CPU EPP is unreadable", "stdout": "", "readback": None,
                                 "duration_us": 0})
                self._epp_initial = initial
            failed = [row for row in rows if not command_succeeded(row)]
            if failed:
                self._record("OFF", "UNAVAILABLE", "capability_probe_failed", rows)
                return False
            restore = self._clock.restore()
            rows.append(restore)
            if not command_succeeded(restore):
                self._record("OFF", "UNAVAILABLE", "command_failed", rows)
                return False
            self._record("OFF", "RESTORED", "capability_probe", rows)
            with self._condition:
                self._track_idle_locked()
            return True

    def start(self) -> None:
        """Start the converging thread; a no-op unless the probe left the controller RESTORED."""
        with self._condition:
            if self._state in _INERT_STATES or self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="device-power-" + self.policy.device, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._stop:
                    return
                self._condition.wait(self._wake_timeout_s)
                if self._stop:
                    return
            self._converge()

    def end_trace(self) -> None:
        """The paid window ends: restore now and never lock again (synchronous)."""
        with self._condition:
            self._inputs.frozen = True
            self._track_idle_locked()
        self._converge("end_trace")
        self._persist_telemetry()

    def close(self) -> None:
        """Stop the thread, then restore: a controlled state issues the ordinary restore; an
        UNAVAILABLE controller whose lock ever succeeded gets one best-effort ``-rgc`` (and EPP
        restore) so the GPU never stays locked after the run. Idempotent."""
        with self._condition:
            self._inputs.frozen = True
            self._track_idle_locked()
            self._stop = True
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join(timeout=max(5.0, 4 * COMMAND_TIMEOUT_S))
            if thread.is_alive():
                raise PhysicalAdapterError("device power controller did not stop")
        with self._condition:
            state = self._state
        if state == "UNAVAILABLE":
            self._best_effort_restore()
        elif state != "OFF":
            self._converge("close")
        self._persist_telemetry()

    def _best_effort_restore(self) -> None:
        with self._command_lock:
            rows = []
            if self._ever_locked and self._gpu_lock is not None:
                rows.append(self._clock.restore())
                self._gpu_lock = None
            if self._epp is not None and self._epp_applied is not None and self._epp_initial is not None:
                rows.append(self._epp.write(self._epp_initial))
                self._epp_applied = None
            if rows:
                self._record("UNAVAILABLE", "UNAVAILABLE", "close_best_effort", rows)

    # ---- inputs (non-blocking) -------------------------------------------------------------

    def _set(self, **changes: object) -> None:
        with self._condition:
            for name, value in changes.items():
                setattr(self._inputs, name, value)
            closed = self._track_idle_locked()
            self._condition.notify_all()
        if closed:
            self._persist_telemetry()

    def note_next_arrival_us(self, arrival_us: int | None) -> None:
        """The next trace arrival on the RESULT clock; None means no further arrival. Refused
        online: that controller never receives a future arrival."""
        if self.policy.online:
            raise PhysicalAdapterError("online device power never takes the next trace arrival")
        if arrival_us is not None and (type(arrival_us) is not int or arrival_us < 0):
            raise PhysicalAdapterError("device power next arrival is invalid")
        self._set(next_arrival_us=arrival_us, arrivals_finished=arrival_us is None)

    def note_arrival_observed(self, request_id: str, model_id: str, observed_at_us: int) -> None:
        """Online only: ``request_id`` of ``model_id`` arrived at ``observed_at_us`` (RESULT clock),
        which must not be later than now and not earlier than the previous observation. The GPU
        stays restored for ``min_gap_s`` or until its work starts."""
        if self._predictor is None:
            raise PhysicalAdapterError("device power arrival observations need arrival_information online")
        for value in (request_id, model_id):
            if type(value) is not str or not value or not value.isascii():
                raise PhysicalAdapterError("device power arrival observation is invalid")
        if type(observed_at_us) is not int or observed_at_us < 0:
            raise PhysicalAdapterError("device power arrival observation time is invalid")
        if observed_at_us > self._at_us():
            raise PhysicalAdapterError("device power arrival observation is in the future")
        with self._condition:
            if self._observed_arrivals and observed_at_us < self._observed_arrivals[-1]["observed_at_us"]:
                raise PhysicalAdapterError("device power arrival observations are out of order")
            self._predictor.observe(model_id, observed_at_us)
            self._observed_arrivals.append({"request_id": request_id, "model_id": model_id,
                                            "observed_at_us": observed_at_us, "noted_at_us": self._at_us()})
            if self._state not in _INERT_STATES:
                self._inputs.pending_arrival_us = observed_at_us
            self._condition.notify_all()
        self._persist_telemetry()

    def note_first_token(self, request_id: str) -> None:
        """``protect_prefill``: ``request_id`` streamed its first token, so its active GPU
        executions are past prompt processing and may be capped. A no-op without the flag."""
        if type(request_id) is not str or not request_id:
            raise PhysicalAdapterError("device power first-token request is invalid")
        if not self._protect_prefill:
            return
        with self._condition:
            for ticket_id, owner in self._inputs.request_by_ticket.items():
                if owner == request_id and ticket_id in self._inputs.active_gpu:
                    self._inputs.decoding.add(ticket_id)
            self._condition.notify_all()

    def note_queued_start_us(self, start_us: int | None) -> None:
        """Earliest planned start of a QUEUED ticket, None when nothing is queued."""
        if start_us is not None and (type(start_us) is not int or start_us < 0):
            raise PhysicalAdapterError("device power queued start is invalid")
        self._set(queued_start_us=start_us)

    def note_transition_active(self, active: bool) -> None:
        if type(active) is not bool:
            raise PhysicalAdapterError("device power transition flag is invalid")
        self._set(transition_active=active)

    def bind_queued_start_provider(self, provider: Callable[[], int | None] | None) -> None:
        """Poll ``provider`` at every convergence for the queued start (outside every lock)."""
        if provider is not None and not callable(provider):
            raise PhysicalAdapterError("device power queued start provider is invalid")
        with self._condition:
            self._queued_start_provider = provider

    def _targets_device(self, command: PhysicalExecutionCommand) -> bool:
        return any(participant.device_id == self.policy.device for participant in command.participants)

    def on_execution_start(self, command: PhysicalExecutionCommand) -> None:
        """A GPU-targeting execution starts: track it and, when the clocks are still lowered
        (IDLE_MIN/LOAD_MIN), restore synchronously (``reason="late_restore"``) before returning;
        a decode cap is applied by the thread. With ``protect_prefill`` a DECODE_CAP is lifted
        synchronously too (``prefill_restore``); with telemetry a command still in flight is
        waited for, and the wait is recorded. Non-GPU commands are ignored."""
        if not isinstance(command, PhysicalExecutionCommand):
            raise PhysicalAdapterError("device power execution command is invalid")
        if not self._targets_device(command):
            return
        started_ns = self._monotonic_ns()
        with self._condition:
            if self._state in _INERT_STATES:
                return
            self._inputs.active_gpu[command.ticket_id] = command.executor_id
            if self._protect_prefill:
                self._inputs.request_by_ticket[command.ticket_id] = command.request_id
                self._inputs.decoding.discard(command.ticket_id)
            self._inputs.pending_arrival_us = None
            previous = self._state
            lowered = previous in ("IDLE_MIN", "LOAD_MIN") or (self._protect_prefill and previous == "DECODE_CAP")
            in_flight = self._telemetry and self._command_lock.locked()
            closed = self._track_idle_locked()
            self._condition.notify_all()
        if lowered or in_flight:
            self._converge("prefill_restore" if previous == "DECODE_CAP" else "late_restore")
        if self._telemetry:
            with self._condition:
                self._execution_waits.append({
                    "ticket_id": command.ticket_id, "request_id": command.request_id,
                    "at_us": self._ns_to_us(started_ns),
                    "wait_us": max(0, (self._monotonic_ns() - started_ns) // 1000),
                    "state_before": previous, "state_after": self._state,
                    "synchronous": lowered or in_flight,
                })
            self._persist_telemetry()
        elif closed:
            self._persist_telemetry()

    def _forget_execution_locked(self, ticket_id: str) -> None:
        self._inputs.active_gpu.pop(ticket_id, None)
        self._inputs.request_by_ticket.pop(ticket_id, None)
        self._inputs.decoding.discard(ticket_id)

    def on_execution_finish(self, command: PhysicalExecutionCommand) -> None:
        if not isinstance(command, PhysicalExecutionCommand):
            raise PhysicalAdapterError("device power execution command is invalid")
        with self._condition:
            self._forget_execution_locked(command.ticket_id)
            closed = self._track_idle_locked()
            self._condition.notify_all()
        if closed:
            self._persist_telemetry()

    def on_server_stopped(self, executor_id: str) -> None:
        """A server stopped: its executions can no longer finish, release their cap."""
        with self._condition:
            for ticket_id in [key for key, value in self._inputs.active_gpu.items() if value == executor_id]:
                self._forget_execution_locked(ticket_id)
            closed = self._track_idle_locked()
            self._condition.notify_all()
        if closed:
            self._persist_telemetry()

    def on_load_begin(self) -> None:
        """A model load starts (old servers stopped, disk-bound until health): synchronous."""
        self._set(load_active=True)
        self._converge("load_begin")

    def on_load_end(self) -> None:
        """The load reached health (or failed): synchronous restore; idempotent."""
        with self._condition:
            was_loading = self._inputs.load_active
            self._inputs.load_active = False
            self._track_idle_locked()
            self._condition.notify_all()
        if was_loading:
            self._converge("load_end")

    # ---- convergence ------------------------------------------------------------------------

    def _target(self, now_us: int) -> tuple[str, str, float]:
        """(target state, reason, seconds until the decision may change) from the inputs."""
        inputs, state, policy = self._inputs, self._state, self.policy
        self._decision = None
        if inputs.active_gpu:
            if self._protect_prefill and any(ticket not in inputs.decoding for ticket in inputs.active_gpu):
                return "RESTORED", "prefill_active", self._tick_interval_s
            return ("DECODE_CAP" if policy.decode_cap is not None else "RESTORED", "execution_active", self._tick_interval_s)
        if inputs.load_active:
            return ("LOAD_MIN" if policy.load_min else "RESTORED", "load", self._tick_interval_s)
        if inputs.frozen:
            return "RESTORED", "trace_ended", self._tick_interval_s
        if policy.idle is None:
            return "RESTORED", "no_idle_policy", self._tick_interval_s
        if inputs.transition_active:
            return "RESTORED", "transition_active", self._tick_interval_s
        if inputs.queued_start_us is not None:
            return "RESTORED", "ticket_queued", self._tick_interval_s
        if self._predictor is not None:
            return self._online_target(now_us)
        if inputs.next_arrival_us is None and not inputs.arrivals_finished:
            return "RESTORED", "arrival_unknown", self._tick_interval_s
        gap_us = None if inputs.arrivals_finished else inputs.next_arrival_us - now_us
        lead_us = policy.idle.lead_ms * 1000
        if state == "IDLE_MIN":
            if gap_us is None or gap_us >= lead_us:
                wait_s = self._tick_interval_s if gap_us is None else min(self._tick_interval_s, (gap_us - lead_us) / 1e6)
                return "IDLE_MIN", "idle_gap", max(0.0, wait_s)
            return "RESTORED", "predictive_restore", self._tick_interval_s
        if gap_us is None or gap_us >= policy.idle.min_gap_s * 1_000_000:
            return "IDLE_MIN", "idle_gap", self._tick_interval_s
        return "RESTORED", "gap_short", self._tick_interval_s

    def _online_target(self, now_us: int) -> tuple[str, str, float]:
        """The idle rule without future arrivals (see the class docstring); records its inputs."""
        inputs, online, tick = self._inputs, self.policy.online_idle, self._tick_interval_s
        horizon_us = self.policy.idle.min_gap_s * 1_000_000
        estimate = self._predictor.estimate(now_us)
        idle_us = None if self._idle_since_ns is None else max(0, (self._monotonic_ns() - self._idle_since_ns) // 1000)
        self._decision = {"estimate": estimate, "idle_us": idle_us, "pending_arrival_us": inputs.pending_arrival_us}
        if estimate is None:
            return "RESTORED", "arrival_unobserved", tick
        pending = inputs.pending_arrival_us
        if pending is not None and now_us - pending < horizon_us:
            return "RESTORED", "arrival_observed", max(0.0, min(tick, (pending + horizon_us - now_us) / 1e6))
        if self._state == "IDLE_MIN":
            return "IDLE_MIN", "idle_hold", tick
        if idle_us is None:
            return "RESTORED", "idle_settling", tick
        waits = [tick]
        likely = estimate["probability_ppm"] > online.max_arrival_probability_ppm
        if not likely:
            if idle_us >= ONLINE_IDLE_SETTLE_US:
                return "IDLE_MIN", "predicted_idle", tick
            waits.append((ONLINE_IDLE_SETTLE_US - idle_us) / 1e6)
        if online.fallback_idle_ms is not None:
            fallback_us = online.fallback_idle_ms * 1000
            if idle_us >= fallback_us:
                return "IDLE_MIN", "idle_timeout", tick
            waits.append((fallback_us - idle_us) / 1e6)
        return "RESTORED", "arrival_likely" if likely else "idle_settling", max(0.0, min(waits))

    def _poll_queued_start(self) -> None:
        with self._condition:
            provider = self._queued_start_provider
        if provider is None:
            return
        value = provider()
        if value is not None and (type(value) is not int or value < 0):
            raise PhysicalAdapterError("device power queued start provider returned an invalid value")
        with self._condition:
            self._inputs.queued_start_us = value
            closed = self._track_idle_locked()
        if closed:
            self._persist_telemetry()

    def _converge(self, reason_override: str | None = None) -> None:
        """Bring the hardware to the target state; commands run under the command lock only."""
        with self._command_lock:
            self._poll_queued_start()
            with self._condition:
                if self._state in _INERT_STATES:
                    return
                target, reason, wait_s = self._target(self._at_us())
                self._wake_timeout_s = wait_s
                previous = self._state
                decision = None if self._predictor is None else copy.deepcopy(self._decision)
            if target == previous:
                return
            results, failed = self._apply(target)
            if failed:
                self._record(previous, "UNAVAILABLE", "command_failed", results, decision)
                return
            self._record(previous, target, reason if reason_override is None else reason_override, results, decision)

    def _apply(self, target: str) -> tuple[list[CommandResult], bool]:
        """Issue the commands that move the hardware to ``target``; stop at the first failure."""
        results: list[CommandResult] = []
        for step in self._steps(target):
            result = step()
            results.append(result)
            if not command_succeeded(result):
                return results, True
        return results, False

    def _steps(self, target: str) -> list[Callable[[], CommandResult]]:
        idle = self.policy.idle
        floor = None if idle is None else idle.gpu_min_clocks_mhz
        lock = {
            "IDLE_MIN": None if floor is None else (floor, floor),
            "LOAD_MIN": None if floor is None else (floor, floor),
            "DECODE_CAP": None if floor is None or self.policy.decode_cap is None else (floor, self.policy.decode_cap.sm_max_mhz),
            "RESTORED": None,
        }[target]
        epp = idle.cpu_epp if target == "IDLE_MIN" and idle is not None else None
        steps: list[Callable[[], CommandResult]] = []
        if lock != self._gpu_lock:
            steps.append(self._restore_gpu if lock is None else (lambda pair=lock: self._lock_gpu(pair)))
        if self._epp is not None and epp != self._epp_applied:
            steps.append(lambda value=epp: self._write_epp(value))
        return steps

    def _lock_gpu(self, pair: tuple[int, int]) -> CommandResult:
        result = self._clock.lock(*pair)
        if command_succeeded(result):
            self._gpu_lock = pair
            self._ever_locked = True
        return result

    def _restore_gpu(self) -> CommandResult:
        result = self._clock.restore()
        if command_succeeded(result):
            self._gpu_lock = None
        return result

    def _write_epp(self, value: str | None) -> CommandResult:
        if value is None:
            if self._epp_initial is None:
                return {"command": ["read", CPU_EPP_FILE], "returncode": None, "stderr": "initial EPP is unknown",
                        "stdout": "", "readback": None, "duration_us": 0}
            result = self._epp.write(self._epp_initial)
        else:
            result = self._epp.write(value)
        if command_succeeded(result):
            self._epp_applied = value
        return result


__all__ = [
    "ArrivalGapPredictor",
    "COMMAND_TIMEOUT_S",
    "CPU_EPP_VALUES",
    "CpuEppControl",
    "DEVICE_POWER_EVENTS_FILE",
    "DEVICE_POWER_EVENT_KIND",
    "DEVICE_POWER_STATES",
    "DEVICE_POWER_TELEMETRY_FILE",
    "DEVICE_POWER_TELEMETRY_SCHEMA",
    "DeviceClockControl",
    "DevicePowerController",
    "DevicePowerPolicy",
    "NvidiaClockControl",
    "command_succeeded",
    "device_power_capability",
    "run_control_command",
]
