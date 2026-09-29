"""Raw HTTP probes used by canonical physical runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from email.utils import parsedate_to_datetime
import http.client
import json
import subprocess
import time
from urllib.parse import urlsplit
from pathlib import Path
import re
import threading
from typing import Callable

from .contracts import PhysicalAdapterError
from .snapshot import EndpointRuntimeSample


@dataclass(frozen=True)
class AndroidProcessIdentity:
    process_id: int
    boot_id: str
    start_ticks: int
    executable: str

    def __post_init__(self) -> None:
        if (type(self.process_id) is not int or self.process_id <= 0
                or type(self.start_ticks) is not int or self.start_ticks <= 0
                or type(self.boot_id) is not str
                or not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", self.boot_id)
                or type(self.executable) is not str or not self.executable.startswith("/")):
            raise PhysicalAdapterError("Android process identity is invalid")

    def to_json(self) -> dict[str, object]:
        return {"process_id": self.process_id, "boot_id": self.boot_id,
                "start_ticks": self.start_ticks, "executable": self.executable}


def parse_android_process_identity(output: str, process_id: int) -> AndroidProcessIdentity:
    rows = output.strip().splitlines()
    if (len(rows) != 4 or not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", rows[0])
            or rows[3] != str(process_id) or not rows[2].startswith("/")):
        raise PhysicalAdapterError("Android process identity is malformed")
    match = re.fullmatch(r"(\d+) \(.*\) (.+)", rows[1])
    fields = () if match is None else match[2].split()
    if (match is None or int(match[1]) != process_id or len(fields) < 20
            or fields[0] in {"Z", "X", "x"} or not fields[19].isdigit() or int(fields[19]) <= 0):
        raise PhysicalAdapterError("Android process lifetime is invalid")
    return AndroidProcessIdentity(process_id, rows[0], int(fields[19]), rows[2])


def parse_android_process_allocation(
    output: str, expected: AndroidProcessIdentity, *, captured_at_ns: int, finished_at_ns: int,
) -> dict[str, object]:
    sections = output.split("\nS42_PROCESS_MEMORY_V1\n")
    if len(sections) != 3:
        raise PhysicalAdapterError("Android allocation observation is malformed")
    before, memory, after = sections
    if (parse_android_process_identity(before, expected.process_id) != expected
            or parse_android_process_identity(after, expected.process_id) != expected):
        raise PhysicalAdapterError("Android allocation process lifetime changed")
    pids = re.findall(r"\*\* MEMINFO in pid (\d+) \[", memory)
    summary = re.findall(r"\bTOTAL PSS:\s*(\d+)\b", memory)
    table = re.findall(r"^\s*TOTAL\s+(\d+)\s+", memory, re.MULTILINE)
    totals = summary + table
    if (pids != [str(expected.process_id)] or not totals or len(summary) > 1 or len(table) > 1
            or len(set(totals)) != 1 or int(totals[0]) <= 0
            or (not summary and not re.search(r"\bPss\b", memory))
            or type(captured_at_ns) is not int or type(finished_at_ns) is not int
            or not 0 < captured_at_ns <= finished_at_ns):
        raise PhysicalAdapterError("Android allocation PSS is missing or malformed")
    return {
        "source": "android-dumpsys-meminfo-pss-v1", "process_identity": expected.to_json(),
        "allocated_bytes": int(totals[0]) * 1024, "captured_at_ns": captured_at_ns,
        "finished_at_ns": finished_at_ns,
        "raw_sha256": "sha256:" + hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def parse_android_process_memory_peak(
    output: str, expected: AndroidProcessIdentity, *, captured_at_ns: int, finished_at_ns: int,
) -> dict[str, object]:
    sections = output.split("\nS42_PROCESS_PEAK_V1\n")
    if len(sections) != 4:
        raise PhysicalAdapterError("Android memory peak observation is malformed")
    before, status, gpu, after = sections
    if (parse_android_process_identity(before, expected.process_id) != expected
            or parse_android_process_identity(after, expected.process_id) != expected):
        raise PhysicalAdapterError("Android memory peak process lifetime changed")
    values = {}
    for name in ("VmRSS", "VmHWM", "VmSwap"):
        matches = re.findall(r"^" + name + r":\s+(\d+) kB$", status, re.MULTILINE)
        if len(matches) != 1:
            raise PhysicalAdapterError("Android memory peak host counter is malformed: " + name)
        values[name] = int(matches[0]) * 1024
    keys = ("kernel", "kernel_max", "user", "user_max", "imported_mem")
    rows = gpu.strip().splitlines()
    if len(rows) != len(keys) or any(not row.isdigit() for row in rows):
        raise PhysicalAdapterError("Android memory peak GPU counters are malformed")
    driver = dict(zip(keys, map(int, rows)))
    if (values["VmHWM"] < values["VmRSS"] or values["VmHWM"] <= 0
            or driver["kernel_max"] < driver["kernel"]
            or driver["user_max"] < driver["user"]
            or type(captured_at_ns) is not int or type(finished_at_ns) is not int
            or not 0 < captured_at_ns <= finished_at_ns):
        raise PhysicalAdapterError("Android memory peak counters are inconsistent")
    # This sum intentionally does not subtract CPU/GPU mapping overlap.
    accounted = values["VmHWM"] + driver["kernel_max"] + driver["user_max"]
    return {
        "source": "android-proc-kgsl-high-water-v1", "process_identity": expected.to_json(),
        "host_rss_bytes": values["VmRSS"], "host_rss_peak_bytes": values["VmHWM"],
        "host_swap_bytes": values["VmSwap"], "gpu_counters_bytes": driver,
        "accounted_peak_bytes": accounted,
        "has_unbounded_accounting": bool(values["VmSwap"] or driver["imported_mem"]),
        "captured_at_ns": captured_at_ns, "finished_at_ns": finished_at_ns,
        "raw_sha256": "sha256:" + hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def probe_nvidia_process_memory_bytes(
    process_id: int, *, timeout_s: float = 2
) -> int | None:
    """Return raw NVIDIA memory attributed to one operating-system process."""
    if (
        type(process_id) is not int
        or process_id <= 0
        or timeout_s <= 0
    ):
        raise PhysicalAdapterError("NVIDIA process probe is invalid")
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    total_mib = 0
    matched = False
    for line in completed.stdout.splitlines():
        fields = tuple(value.strip() for value in line.split(","))
        if len(fields) != 2:
            return None
        try:
            observed_pid = int(fields[0])
            used_mib = int(fields[1])
        except ValueError:
            return None
        if observed_pid != process_id:
            continue
        if used_mib <= 0:
            return None
        matched = True
        total_mib += used_mib
    return total_mib * 1024 * 1024 if matched else None


def _endpoint(endpoint: str) -> tuple[str, int]:
    value = urlsplit(endpoint)
    if (
        value.scheme != "http"
        or value.hostname is None
        or value.port is None
        or value.query
        or value.fragment
    ):
        raise PhysicalAdapterError("runtime probe endpoint is invalid")
    return value.hostname, value.port


def http_json(
    endpoint: str, path: str, timeout_s: float,
    *, response_headers: dict[str, str | None] | None = None,
) -> object:
    host, port = _endpoint(endpoint)
    if (
        type(path) is not str
        or not path.startswith("/")
        or timeout_s <= 0
    ):
        raise PhysicalAdapterError("runtime HTTP probe is invalid")
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise PhysicalAdapterError(
                "runtime HTTP probe status is invalid: " + str(response.status)
            )
        if response_headers is not None:
            response_headers.update({
                name: response.getheader(name) for name in ("Date", "Age")
            })
        return json.loads(payload)
    finally:
        connection.close()


def probe_llama_endpoint(endpoint: str) -> EndpointRuntimeSample:
    """Report endpoint health and slots without making eligibility decisions."""
    try:
        health = http_json(endpoint, "/health", 1)
    except (OSError, ValueError, json.JSONDecodeError, PhysicalAdapterError):
        return EndpointRuntimeSample("unavailable", "unavailable", 0)
    if type(health) is not dict or health.get("status") != "ok":
        return EndpointRuntimeSample("unavailable", "unavailable", 0)
    try:
        slots = http_json(endpoint, "/slots", 1)
    except (OSError, ValueError, json.JSONDecodeError, PhysicalAdapterError):
        return EndpointRuntimeSample("healthy", "failed", 0)
    if type(slots) is not list or any(type(row) is not dict for row in slots):
        return EndpointRuntimeSample("healthy", "failed", 0)
    return EndpointRuntimeSample(
        "healthy",
        "live",
        sum(row.get("is_processing") is False for row in slots),
    )


@dataclass(frozen=True)
class PhoneRuntimeProbe:
    capacity_bytes: int
    available_bytes: int
    temperature_millic: int
    battery_ppm: int
    thermal_qualified: bool
    task_server_alive: bool
    charging: bool | None = None
    # raw Android thermal status (0..6); the ADB probe reports the coarsest of the
    # platform status and every physical sensor's throttling status
    thermal_status: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.capacity_bytes) is not int
            or self.capacity_bytes <= 0
            or type(self.available_bytes) is not int
            or not 0 <= self.available_bytes <= self.capacity_bytes
            or type(self.temperature_millic) is not int
            or self.temperature_millic <= 0
            or type(self.battery_ppm) is not int
            or not 0 <= self.battery_ppm <= 1_000_000
            or type(self.thermal_qualified) is not bool
            or type(self.task_server_alive) is not bool
            or (
                self.charging is not None
                and type(self.charging) is not bool
            )
            or (
                self.thermal_status is not None
                and (type(self.thermal_status) is not int or self.thermal_status < 0)
            )
        ):
            raise PhysicalAdapterError("phone runtime probe is invalid")

    def thermal_qualified_under(self, maximum_thermal_status: int) -> bool:
        """Probe verdict under a device's Android thermal status limit.

        The raw status qualifies while it does not exceed the limit; a probe
        without one keeps its own ``thermal_qualified`` verdict.
        """
        if self.thermal_status is not None:
            return self.thermal_status <= maximum_thermal_status
        return self.thermal_qualified


@dataclass(frozen=True)
class PhoneRuntimeObservation:
    source: str
    captured_at_ns: int | None
    checked_at_ns: int
    validity: str
    failure_reason: str | None
    value: PhoneRuntimeProbe | None = None
    attempts: tuple[dict[str, object], ...] = ()
    temperature_source: str | None = None
    freshness_basis: str | None = None

    def to_json(self, now_ns: int | None = None) -> dict[str, object]:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        age_us = (
            None if self.captured_at_ns is None else
            max(0, (now_ns - self.captured_at_ns) // 1000)
        )
        validity, reason = self.validity, self.failure_reason
        if validity == "VALID" and (age_us is None or age_us > 5_000_000):
            validity, reason = "STALE", "phone sample exceeds 5000000 us"
        value = self.value if validity == "VALID" else None
        return {
            "source": self.source,
            "temperature_source": self.temperature_source,
            "sample_timestamp_ns": self.captured_at_ns,
            "checked_at_ns": self.checked_at_ns,
            "age_us": age_us,
            "maximum_age_us": 5_000_000,
            "validity": validity,
            "valid": validity == "VALID",
            "failure_reason": reason,
            "attempts": list(self.attempts),
            **({"freshness_basis": self.freshness_basis}
               if self.freshness_basis is not None else {}),
            "available_bytes": None if value is None else value.available_bytes,
            "temperature_millic": None if value is None else value.temperature_millic,
            "battery_ppm": None if value is None else value.battery_ppm,
        }


class _PhoneProbeFailure(ValueError):
    def __init__(self, validity: str, reason: str) -> None:
        super().__init__(reason)
        self.validity = validity


def _runtime_probe_result(
    source, started_ns, diagnostic, value=None, *, validity="VALID",
    reason=None, sampled_at_ns=None, temperature_source=None, freshness_basis=None,
):
    if not diagnostic:
        return value
    return PhoneRuntimeObservation(
        source=source,
        captured_at_ns=(
            sampled_at_ns if sampled_at_ns is not None else
            started_ns if value is not None else None
        ),
        checked_at_ns=time.monotonic_ns(),
        validity=validity,
        failure_reason=reason,
        value=value,
        temperature_source=temperature_source,
        freshness_basis=freshness_basis,
    )


def _phone_sample_timestamp(captured_epoch_s, headers, started_ns, checked_ns):
    if type(captured_epoch_s) is not int:
        raise _PhoneProbeFailure("MALFORMED", "invalid captured_epoch_s")
    date = headers.get("Date")
    if date is None:
        age_s = time.time() - captured_epoch_s
        if abs(age_s) > 5:
            raise _PhoneProbeFailure("STALE", "snapshot clock age_s=" + str(age_s))
        return checked_ns - round(age_s * 1e9)
    try:
        reference = parsedate_to_datetime(date)
        if reference.tzinfo is None:
            raise ValueError("HTTP Date has no timezone")
        cache_age = headers.get("Age") or "0"
        if not cache_age.isascii() or not cache_age.isdecimal():
            raise ValueError("HTTP Age is not a nonnegative integer")
        delta_s = int(reference.timestamp()) - captured_epoch_s
        if delta_s < -1:
            raise _PhoneProbeFailure("STALE", "sample is newer than phone HTTP Date")
        # Date/sample seconds are rounded down. Include quantization and the entire RPC.
        age_ns = (max(0, delta_s + 1) + int(cache_age)) * 1_000_000_000
        age_ns += checked_ns - started_ns
    except _PhoneProbeFailure:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise _PhoneProbeFailure("MALFORMED", "invalid phone HTTP clock: " + str(error)) from error
    if checked_ns < started_ns or age_ns > 5_000_000_000:
        raise _PhoneProbeFailure("STALE", "phone response age bound_ns=" + str(age_ns))
    return checked_ns - age_ns


def probe_phone_runtime(
    endpoint: str, *, diagnostic: bool = False,
) -> PhoneRuntimeProbe | PhoneRuntimeObservation | None:
    started_ns = time.monotonic_ns()
    source = endpoint + "/snapshot.json"
    sampled_at_ns = None
    temperature_source = None
    headers: dict[str, str | None] = {}
    freshness_basis = None
    try:
        value = http_json(endpoint, "/snapshot.json", 1, response_headers=headers)
        checked_ns = time.monotonic_ns()
        freshness_basis = "phone-http-date" if headers.get("Date") is not None else "host-wall-clock"
        if type(value) is not dict:
            raise _PhoneProbeFailure("MALFORMED", "snapshot is not an object")
        temperature_source = value.get("temperature_source", "sysfs")
        if type(temperature_source) is not str or temperature_source not in {"sysfs", "android-hal"}:
            raise _PhoneProbeFailure("MALFORMED", "unknown temperature source")
        required = (
            "captured_epoch_s", "mem_available_kib", "mem_total_kib",
            "temperature_max_millic", "battery_level_pct",
            "android_thermal_status", "task_server_alive", "schema",
        )
        missing = tuple(name for name in required if name not in value)
        if missing:
            raise _PhoneProbeFailure("MISSING", "missing fields: " + ",".join(missing))
        captured_epoch_s = value.get("captured_epoch_s")
        available_kib = value.get("mem_available_kib")
        total_kib = value.get("mem_total_kib")
        temperature = value.get("temperature_max_millic")
        battery_level = value.get("battery_level_pct")
        thermal = value.get("android_thermal_status")
        charging = value.get("charging", value.get("usb_online"))
        checks = {
            "schema": value.get("schema") == "s42-op15-live-snapshot-v1",
            "captured_epoch_s": type(captured_epoch_s) is int,
            "mem_total_kib": type(total_kib) is int and total_kib > 0,
            "mem_available_kib": (
                type(available_kib) is int and type(total_kib) is int
                and 0 <= available_kib <= total_kib
            ),
            "temperature_max_millic": type(temperature) is int and temperature > 0,
            "battery_level_pct": type(battery_level) is int and 0 <= battery_level <= 100,
            "android_thermal_status": type(thermal) is int and thermal >= -1,
            "task_server_alive": type(value.get("task_server_alive")) is bool,
        }
        invalid = [name for name, valid in checks.items() if not valid]
        if invalid:
            raise _PhoneProbeFailure("MALFORMED", "invalid fields: " + ",".join(invalid))
        if thermal == -1:
            raise _PhoneProbeFailure("MISSING", "Android thermal status is unknown (-1)")
        sampled_at_ns = _phone_sample_timestamp(
            captured_epoch_s, headers, started_ns, checked_ns
        )
        observed = PhoneRuntimeProbe(
            capacity_bytes=total_kib * 1024,
            available_bytes=available_kib * 1024,
            temperature_millic=temperature,
            battery_ppm=battery_level * 10_000,
            thermal_qualified=thermal == 0,
            task_server_alive=value["task_server_alive"],
            charging=(charging if type(charging) is bool else None),
            thermal_status=thermal,
        )
    except (TimeoutError, subprocess.TimeoutExpired) as error:
        validity, reason = "TIMED_OUT", str(error) or type(error).__name__
    except _PhoneProbeFailure as error:
        validity, reason = error.validity, str(error)
    except (ValueError, UnicodeError) as error:
        validity, reason = "MALFORMED", type(error).__name__ + ": " + str(error)
    except (OSError, http.client.HTTPException, PhysicalAdapterError) as error:
        validity, reason = "UNAVAILABLE", type(error).__name__ + ": " + str(error)
    else:
        return _runtime_probe_result(
            source, started_ns, diagnostic, observed,
            sampled_at_ns=sampled_at_ns,
            temperature_source=temperature_source,
            freshness_basis=freshness_basis,
        )
    return _runtime_probe_result(
        source, started_ns, diagnostic, validity=validity,
        reason=reason, sampled_at_ns=sampled_at_ns,
        temperature_source=temperature_source,
        freshness_basis=freshness_basis,
    )


_ANDROID_TEMPERATURE = re.compile(
    r"Temperature\{mValue=([-+]?[0-9]+(?:\.[0-9]+)?),"
    r" mType=([0-9]+), mName=([^,}]+), mStatus=([0-9]+)\}"
)

_ANDROID_PHYSICAL_TEMPERATURE_TYPES = frozenset({0, 1, 2, 3, 4, 5, 9})


def probe_android_phone_runtime(
    serial: str, adb_port: int, *, timeout_s: float = 3,
    diagnostic: bool = False,
) -> PhoneRuntimeProbe | PhoneRuntimeObservation | None:
    """Read memory and thermal state through an authorized ADB endpoint."""
    if (
        type(serial) is not str
        or not serial
        or not serial.isascii()
        or any(character.isspace() for character in serial)
        or type(adb_port) is not int
        or not 0 < adb_port <= 65_535
        or timeout_s <= 0
    ):
        raise PhysicalAdapterError("Android runtime probe setup is invalid")
    started_ns = time.monotonic_ns()
    source = "adb:" + serial + ":" + str(adb_port)

    def failed(validity, reason):
        return _runtime_probe_result(
            source, started_ns, diagnostic, validity=validity, reason=reason,
        )
    command = (
        "cat /proc/meminfo; dumpsys thermalservice; "
        "dumpsys battery | awk '"
        "/^[[:space:]]*level:/{level=$2} "
        "/^[[:space:]]*USB powered:/{usb=$3} "
        "/^[[:space:]]*status:/{status=$2} "
        "END{print \"S42_BATTERY_LEVEL=\" level; "
        "print \"S42_BATTERY_CHARGING=\" "
        "((usb==\"true\" || status==2 || status==5) ? 1 : 0)}'; "
        "printf '\\nS42_TASK_SERVER='; pidof llama-server || true"
    )
    try:
        completed = subprocess.run(
            [
                "adb", "-P", str(adb_port), "-s", serial,
                "shell", command,
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        return failed("TIMED_OUT", "ADB shell exceeded " + str(error.timeout) + " s")
    except OSError as error:
        return failed("UNAVAILABLE", type(error).__name__ + ": " + str(error))
    if completed.returncode != 0:
        return failed("UNAVAILABLE", "ADB exit " + str(completed.returncode) + ": " + completed.stderr.strip())
    memory = {}
    for name in ("MemTotal", "MemAvailable"):
        match = re.search(r"^" + name + r":\s+([0-9]+) kB$", completed.stdout, re.M)
        if match is not None:
            memory[name] = int(match.group(1)) * 1024
    status = re.search(
        r"^Thermal Status:\s+([0-9]+)$", completed.stdout, re.M
    )
    current = completed.stdout.partition("Current temperatures from HAL:")[2]
    current = current.partition("Current cooling devices from HAL:")[0]
    temperatures = tuple(
        (round(float(value) * 1000), int(sensor_status))
        for value, sensor_type, _sensor_name, sensor_status
        in _ANDROID_TEMPERATURE.findall(current)
        if (
            float(value) > 0
            and int(sensor_type) in _ANDROID_PHYSICAL_TEMPERATURE_TYPES
        )
    )
    if (
        set(memory) != {"MemTotal", "MemAvailable"}
        or status is None
        or not temperatures
    ):
        return failed("MISSING", "ADB memory, thermal status, or current physical temperatures are absent")
    if not 0 <= memory["MemAvailable"] <= memory["MemTotal"] or memory["MemTotal"] <= 0:
        return failed("MALFORMED", "ADB memory values are invalid")
    task_server = re.search(
        r"^S42_TASK_SERVER=(.*)$", completed.stdout, re.M
    )
    battery_level = re.search(
        r"^S42_BATTERY_LEVEL=([0-9]+)$", completed.stdout, re.M
    )
    battery_charging = re.search(
        r"^S42_BATTERY_CHARGING=([01])$", completed.stdout, re.M
    )
    if (
        task_server is None
        or battery_level is None
    ):
        return failed("MISSING", "ADB task-server or battery fields are absent")
    if not 0 <= int(battery_level.group(1)) <= 100:
        return failed("MALFORMED", "ADB battery level is invalid")
    thermal_status = max(
        int(status.group(1)),
        *(sensor_status for _, sensor_status in temperatures),
    )
    observed = PhoneRuntimeProbe(
        capacity_bytes=memory["MemTotal"],
        available_bytes=memory["MemAvailable"],
        temperature_millic=max(value for value, _ in temperatures),
        battery_ppm=int(battery_level.group(1)) * 10_000,
        thermal_qualified=thermal_status == 0,
        task_server_alive=bool(task_server.group(1).strip()),
        charging=(
            None
            if battery_charging is None
            else battery_charging.group(1) == "1"
        ),
        thermal_status=thermal_status,
    )
    return _runtime_probe_result(
        source, started_ns, diagnostic, observed, temperature_source="android-hal",
    )


def probe_phone_runtime_with_adb_fallback(
    endpoint: str, serial: str, adb_port: int, *, diagnostic: bool = False,
    fallback_serial_provider: Callable[[], str] | None = None,
) -> PhoneRuntimeProbe | PhoneRuntimeObservation | None:
    options = {"diagnostic": True} if diagnostic else {}
    observed = probe_phone_runtime(endpoint, **options)
    if not diagnostic and observed is not None:
        return observed
    if diagnostic and observed.to_json()["valid"]:
        return observed
    try:
        fallback_serial = serial if fallback_serial_provider is None else fallback_serial_provider()
        fallback = probe_android_phone_runtime(fallback_serial, adb_port, **options)
    except (PhysicalAdapterError, OSError, subprocess.SubprocessError) as error:
        fallback = _runtime_probe_result(
            "adb-control", time.monotonic_ns(), diagnostic,
            validity="TIMED_OUT" if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) else "UNAVAILABLE",
            reason=str(error),
        )
    if not diagnostic:
        return fallback
    return PhoneRuntimeObservation(
        source=fallback.source,
        captured_at_ns=fallback.captured_at_ns,
        checked_at_ns=fallback.checked_at_ns,
        validity=fallback.validity,
        failure_reason=fallback.failure_reason,
        value=fallback.value,
        attempts=(observed.to_json(), fallback.to_json()),
    )


@dataclass(frozen=True)
class LinuxHostRuntimeSample:
    memory_total_bytes: int
    memory_available_bytes: int
    cpu_utilization_pct: int
    memory_stall_avg10_basis_points: int


class LinuxHostRuntimeProbe:
    """Collect raw Linux CPU, memory, and memory-pressure observations."""

    def __init__(
        self,
        *,
        proc_stat: Path = Path("/proc/stat"),
        proc_meminfo: Path = Path("/proc/meminfo"),
        proc_memory_pressure: Path = Path("/proc/pressure/memory"),
    ) -> None:
        self._proc_stat = proc_stat
        self._proc_meminfo = proc_meminfo
        self._proc_memory_pressure = proc_memory_pressure
        self._previous: tuple[int, int] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _cpu_row(text: str) -> tuple[int, int]:
        fields = text.splitlines()[0].split()
        if len(fields) < 8 or fields[0] != "cpu":
            raise PhysicalAdapterError("Linux CPU observation is invalid")
        values = tuple(int(value) for value in fields[1:])
        total = sum(values)
        idle = values[3] + values[4]
        return total, idle

    @staticmethod
    def _memory(text: str) -> tuple[int, int]:
        values = {}
        for line in text.splitlines():
            name, separator, raw = line.partition(":")
            if separator and name in {"MemTotal", "MemAvailable"}:
                fields = raw.strip().split()
                if len(fields) != 2 or fields[1] != "kB":
                    raise PhysicalAdapterError(
                        "Linux memory observation is invalid"
                    )
                values[name] = int(fields[0]) * 1024
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", 0)
        if not 0 < available <= total:
            raise PhysicalAdapterError(
                "Linux memory observation is invalid"
            )
        return total, available

    @staticmethod
    def _pressure(text: str) -> int:
        rows = [line for line in text.splitlines() if line.startswith("some ")]
        if len(rows) != 1:
            raise PhysicalAdapterError(
                "Linux memory pressure observation is invalid"
            )
        fields = dict(
            item.split("=", 1)
            for item in rows[0].split()[1:]
            if "=" in item
        )
        try:
            return max(0, round(float(fields["avg10"]) * 100))
        except (KeyError, ValueError) as error:
            raise PhysicalAdapterError(
                "Linux memory pressure observation is invalid"
            ) from error

    def sample(self) -> LinuxHostRuntimeSample:
        total, idle = self._cpu_row(
            self._proc_stat.read_text(encoding="ascii")
        )
        memory_total, memory_available = self._memory(
            self._proc_meminfo.read_text(encoding="ascii")
        )
        pressure = self._pressure(
            self._proc_memory_pressure.read_text(encoding="ascii")
        )
        with self._lock:
            previous = self._previous
            self._previous = (total, idle)
        if previous is None or total <= previous[0]:
            utilization = 0
        else:
            elapsed = total - previous[0]
            idle_elapsed = max(0, idle - previous[1])
            utilization = max(
                0, min(100, round(100 * (elapsed - idle_elapsed) / elapsed))
            )
        return LinuxHostRuntimeSample(
            memory_total_bytes=memory_total,
            memory_available_bytes=memory_available,
            cpu_utilization_pct=utilization,
            memory_stall_avg10_basis_points=pressure,
        )


def _phone_power_observation(
    value: object, before_ns: int, after_ns: int
) -> dict[str, int] | None:
    if type(value) is not dict:
        return None
    uptime_s = value.get("uptime_s")
    names = (
        "battery_charge_counter_uah",
        "battery_current_ma",
        "battery_voltage_uv",
        "usb_current_ua",
        "usb_voltage_uv",
    )
    fields = {name: value.get(name) for name in names}
    if not (
        value.get("schema") == "s42-op15-live-power-v1"
        and type(uptime_s) in {int, float}
        and not isinstance(uptime_s, bool)
        and uptime_s > 0
        and all(type(row) is int for row in fields.values())
        and fields["usb_current_ua"] >= 0
        and fields["usb_voltage_uv"] > 0
        and fields["battery_voltage_uv"] > 0
    ):
        return None
    charging = value.get("charging", value.get("usb_online"))
    if type(charging) is not bool:
        charging = fields["usb_current_ua"] > 0
    return {
        **fields,
        "battery_discharge_power_mw": round(
            max(0, fields["battery_current_ma"])
            * fields["battery_voltage_uv"] / 1_000_000
        ),
        "host_sample_t_ns": (before_ns + after_ns) // 2,
        "phone_uptime_ns": round(float(uptime_s) * 1_000_000_000),
        "usb_input_power_mw": round(
            fields["usb_current_ua"]
            * fields["usb_voltage_uv"] / 1_000_000_000
        ),
        "charging": int(charging),
    }


def probe_phone_power(endpoint: str) -> dict[str, int] | None:
    before_ns = time.monotonic_ns()
    headers: dict[str, str | None] = {}
    try:
        value = http_json(endpoint, "/power.json", 1, response_headers=headers)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        PhysicalAdapterError,
    ):
        return None
    after_ns = time.monotonic_ns()
    captured_epoch_s = (
        None if type(value) is not dict else value.get("captured_epoch_s")
    )
    try:
        sampled_at_ns = _phone_sample_timestamp(
            captured_epoch_s, headers, before_ns, after_ns)
    except _PhoneProbeFailure:
        return None
    return _phone_power_observation(value, sampled_at_ns, sampled_at_ns)


def probe_phone_power_history(
    endpoint: str,
) -> tuple[dict[str, int], ...]:
    """Read phone-local power history without assigning host timestamps."""
    try:
        host, port = _endpoint(endpoint)
        connection = http.client.HTTPConnection(host, port, timeout=3)
        try:
            connection.request("GET", "/power.jsonl")
            response = connection.getresponse()
            payload = response.read()
            if response.status != 200:
                return ()
        finally:
            connection.close()
        result = []
        for raw in payload.splitlines():
            value = json.loads(raw)
            observed = _phone_power_observation(value, 0, 0)
            if observed is None:
                return ()
            observed.pop("host_sample_t_ns")
            result.append(observed)
        if not result:
            return ()
        return tuple(result)
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        PhysicalAdapterError,
    ):
        return ()


def probe_android_phone_power(
    serial: str, adb_port: int, *, timeout_s: float = 2
) -> dict[str, int] | None:
    if (
        type(serial) is not str
        or not serial
        or not serial.isascii()
        or any(character.isspace() for character in serial)
        or type(adb_port) is not int
        or not 0 < adb_port <= 65_535
        or timeout_s <= 0
    ):
        raise PhysicalAdapterError("Android power probe setup is invalid")
    sensor_paths = (
        "/sys/class/power_supply/usb/current_now",
        "/sys/class/power_supply/usb/voltage_now",
        "/sys/class/power_supply/battery/current_now",
        "/sys/class/power_supply/battery/voltage_now",
        "/sys/class/power_supply/battery/charge_counter",
    )
    shell_command = (
        "su -c 'cut -d \" \" -f 1 /proc/uptime; cat "
        + " ".join(sensor_paths)
        + "'"
    )
    before_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(
            [
                "adb", "-P", str(adb_port), "-s", serial,
                "shell", shell_command,
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    after_ns = time.monotonic_ns()
    if completed.returncode != 0:
        return None
    fields = completed.stdout.split()
    if len(fields) != 6:
        return None
    try:
        value = {
            "schema": "s42-op15-live-power-v1",
            "uptime_s": float(fields[0]),
            "battery_charge_counter_uah": int(fields[5]),
            "battery_current_ma": int(fields[3]),
            "battery_voltage_uv": int(fields[4]),
            "usb_current_ua": int(fields[1]),
            "usb_voltage_uv": int(fields[2]),
        }
    except ValueError:
        return None
    return _phone_power_observation(value, before_ns, after_ns)


def probe_phone_power_with_adb_fallback(
    endpoint: str, serial: str, adb_port: int, *,
    fallback_serial_provider: Callable[[], str] | None = None,
) -> dict[str, int] | None:
    observed = probe_phone_power(endpoint)
    if observed is not None:
        return observed
    try:
        fallback_serial = serial if fallback_serial_provider is None else fallback_serial_provider()
        return probe_android_phone_power(fallback_serial, adb_port)
    except (PhysicalAdapterError, OSError, subprocess.SubprocessError):
        return None
