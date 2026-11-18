"""The phone driver interface of the portable path, and what every phone can report read-only.

A phone is one ``SitePhone`` row plus a driver chosen by its transport. Every driver offers the same few
operations (``PhoneDriver``); what differs between an NPU phone on its direct USB link and an unknown
phone behind ``adb forward`` stays inside the driver. A new phone needs a site row and, at most, a new
driver registered here -- never an edit of a campaign or of the scheduler.

``read_phone_health`` and ``read_phone_facts`` are plain adb reads (no root needed for the state itself,
root only to see root-owned processes); they change nothing on the phone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

from ..adapters.contracts import PhysicalAdapterError
from ..configuration.site import SiteManifest, SitePaths, SitePhone
from .adb import AdbGuard


WORKER_NAME = "llama-ffn-split-worker"
RESIDENT_WORKERS_NAME = "llama-ffn-split-resident-workers"
RESIDENT_ROUTER_NAME = "llama-ffn-split-resident-router"
# processes that mean "someone runs an LLM workload on this phone"
WORKLOAD_PROCESS = re.compile(r"(^|/)(llama|ffn)[^/ ]*")
MAXIMUM_BATTERY_DECIC = 350          # metered runs start below 35.0 C
# Snapdragon SoC -> Hexagon architecture of its NPU library (a hint for the doctor, never a decision)
HEXAGON_ARCH_HINTS = MappingProxyType({"SM8550": "v73", "SM8650": "v75", "SM8750": "v79", "SM8850": "v81"})


@dataclass(frozen=True)
class HelperShard:
    """One resident shard of a helper: a phone session holding a column suffix of some layers."""

    session_id: str
    remote_path: str
    sha256: str
    layers: tuple[int, ...]
    columns: int
    bytes: int

    def to_json(self) -> dict[str, object]:
        return {"bytes": self.bytes, "columns": self.columns, "layers": list(self.layers),
                "remote_path": self.remote_path, "session_id": self.session_id, "sha256": self.sha256}

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "HelperShard":
        return cls(str(value["session_id"]), str(value["remote_path"]), str(value["sha256"]),
                   tuple(int(layer) for layer in value["layers"]), int(value["columns"]), int(value["bytes"]))


@dataclass(frozen=True)
class HelperBinding:
    """What a helper serves in one run: the model's identity and geometry, and the shards on the phone."""

    artifact_sha256: str
    n_embd: int
    n_ff: int
    activation: str
    shards: tuple[HelperShard, ...]
    pins: Mapping[str, str] = MappingProxyType({})       # phone path -> sha256 (binaries, libraries, scripts)

    def __post_init__(self) -> None:
        if not self.shards:
            raise PhysicalAdapterError("helper binding has no shard")
        if len({shard.columns for shard in self.shards}) != 1:
            raise PhysicalAdapterError("the shards of one helper must store the same columns")
        layers = [layer for shard in self.shards for layer in shard.layers]
        if len(set(layers)) != len(layers) or not layers:
            raise PhysicalAdapterError("helper shards must hold distinct layers")
        if self.activation not in ("swiglu", "geglu"):
            raise PhysicalAdapterError("helper activation must be swiglu or geglu")

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted(layer for shard in self.shards for layer in shard.layers))

    @property
    def stored_columns(self) -> int:
        return self.shards[0].columns

    @property
    def layer_mask(self) -> int:
        return sum(1 << layer for layer in self.layers)

    def to_json(self) -> dict[str, object]:
        return {"activation": self.activation, "artifact_sha256": self.artifact_sha256, "n_embd": self.n_embd,
                "n_ff": self.n_ff, "pins": dict(sorted(self.pins.items())),
                "shards": [shard.to_json() for shard in self.shards]}

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "HelperBinding":
        return cls(str(value["artifact_sha256"]), int(value["n_embd"]), int(value["n_ff"]),
                   str(value["activation"]), tuple(HelperShard.from_json(row) for row in value["shards"]),
                   MappingProxyType({str(k): str(v) for k, v in dict(value.get("pins", {})).items()}))


@dataclass(frozen=True)
class HelperEndpoint:
    """A started helper as the host server sees it."""

    server_environment: Mapping[str, str]      # the S41_SERVER_FFN_* link variables for llama-server
    layers: tuple[int, ...]
    stored_columns: int
    transport: str


@dataclass(frozen=True)
class PhoneHealth:
    awake: bool | None
    cpu_caps_khz: Mapping[str, int]            # cpufreq policy -> scaling_max_freq
    cpu_maximum_khz: Mapping[str, int]         # cpufreq policy -> cpuinfo_max_freq
    battery_decic: int | None
    battery_level: int | None
    charging: bool | None
    thermal_status: int | None
    memory_available_bytes: int | None
    data_free_bytes: int | None
    workload_processes: tuple[str, ...]        # "pid name": LLM workloads running on the phone, anyone's

    @property
    def capped(self) -> bool:
        """Some CPU cluster is held below its hardware maximum (a sleeping phone caps its clocks)."""
        return any(cap < self.cpu_maximum_khz.get(policy, cap) for policy, cap in self.cpu_caps_khz.items())

    def to_json(self) -> dict[str, object]:
        return {"awake": self.awake, "battery_decic": self.battery_decic, "battery_level": self.battery_level,
                "charging": self.charging, "cpu_caps_khz": dict(sorted(self.cpu_caps_khz.items())),
                "cpu_maximum_khz": dict(sorted(self.cpu_maximum_khz.items())),
                "data_free_bytes": self.data_free_bytes, "memory_available_bytes": self.memory_available_bytes,
                "thermal_status": self.thermal_status, "workload_processes": list(self.workload_processes)}


class PhoneDriver(Protocol):
    phone: SitePhone

    def describe(self) -> Mapping[str, object]: ...                      # read-only facts
    def health(self) -> PhoneHealth: ...                                 # read-only state
    def verify(self, binding: HelperBinding) -> Mapping[str, str]: ...   # observed sha256; raises on a mismatch
    def push(self, local: Path, remote: str, sha256: str) -> str: ...    # only under this site's directories
    def start(self, binding: HelperBinding, log_directory: Path) -> HelperEndpoint: ...
    def alive(self) -> bool: ...                                         # never called inside a metered phase
    def stop(self, *, served_calls: int) -> Mapping[str, object]: ...
    def release(self) -> Sequence[str]: ...                              # failure path: what could not be cleaned
    def power_calibration(self): ...                                     # PhonePowerCalibration | None


_DRIVERS: dict[str, Callable[[SitePhone, SitePaths, AdbGuard], PhoneDriver]] = {}


def register_phone_driver(transport: str, factory: Callable[[SitePhone, SitePaths, AdbGuard], PhoneDriver]) -> None:
    _DRIVERS[transport] = factory


def phone_driver(site: SiteManifest, device_id: str, *, adb: AdbGuard | None = None) -> PhoneDriver:
    """The driver of one phone of the site, by its transport."""
    phone = site.phone(device_id)
    if phone.transport not in _DRIVERS:
        raise PhysicalAdapterError(f"no portable driver for transport {phone.transport} "
                                   f"(known: {','.join(sorted(_DRIVERS)) or 'none'})")
    return _DRIVERS[phone.transport](phone, site.paths, adb or AdbGuard(site.paths.adb_path, phone.adb_port))


def allowed_phone_roots(phone: SitePhone) -> tuple[str, ...]:
    """The only directories on a phone this site writes to."""
    return (phone.work_directory, phone.bundle_directory)


# --------------------------------------------------------------------------------------------------
# read-only reads
# --------------------------------------------------------------------------------------------------

_HEALTH_COMMAND = (
    "grep -E '^(MemTotal|MemAvailable):' /proc/meminfo; "
    "dumpsys thermalservice 2>/dev/null | grep -m1 'Thermal Status'; "
    "dumpsys battery 2>/dev/null | grep -E '^ *(level|temperature|status|USB powered|AC powered|Wireless powered):'; "
    "dumpsys power 2>/dev/null | grep -m1 'mWakefulness='; "
    "for p in /sys/devices/system/cpu/cpufreq/policy*; do "
    "echo \"S43_CPU ${p##*/} $(cat $p/scaling_max_freq 2>/dev/null) $(cat $p/cpuinfo_max_freq 2>/dev/null)\"; done; "
    "echo \"S43_DATA $(df -k /data 2>/dev/null | tail -1)\""
)
_PROCESS_COMMAND = "ps -A -o PID,NAME"


def _field(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, re.M)
    return None if match is None else match.group(1)


def parse_phone_health(state: str, processes: str) -> PhoneHealth:
    """The output of the health and process commands -> ``PhoneHealth`` (missing parts are None)."""
    def integer(pattern: str) -> int | None:
        value = _field(state, pattern)
        return int(value) if value is not None and re.fullmatch(r"-?[0-9]+", value) else None

    available = integer(r"^MemAvailable:\s+([0-9]+) kB")
    wakefulness = _field(state, r"mWakefulness=(\w+)")
    powered = re.findall(r"^ *(?:USB|AC|Wireless) powered:\s*(\w+)", state, re.M)
    status = integer(r"^ *status:\s*([0-9]+)")
    caps, maxima = {}, {}
    for policy, cap, maximum in re.findall(r"^S43_CPU (\S+) ([0-9]+) ([0-9]+)$", state, re.M):
        caps[policy], maxima[policy] = int(cap), int(maximum)
    free = None
    data = _field(state, r"^S43_DATA (.*)$")
    if data:
        numbers = [word for word in data.split() if word.isdigit()]
        if len(numbers) >= 3:
            free = int(numbers[2]) * 1024            # df -k: total, used, available
    workloads = []
    for line in processes.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].isdigit() and WORKLOAD_PROCESS.search(fields[1]):
            workloads.append(fields[0] + " " + fields[1])
    return PhoneHealth(
        awake=None if wakefulness is None else wakefulness == "Awake",
        cpu_caps_khz=MappingProxyType(caps), cpu_maximum_khz=MappingProxyType(maxima),
        battery_decic=integer(r"^ *temperature:\s*(-?[0-9]+)"), battery_level=integer(r"^ *level:\s*([0-9]+)"),
        charging=None if not powered and status is None else ("true" in powered or status in (2, 5)),
        thermal_status=integer(r"Thermal Status:\s*([0-9]+)"),
        memory_available_bytes=None if available is None else available * 1024,
        data_free_bytes=free, workload_processes=tuple(workloads))


def read_phone_health(adb: AdbGuard, phone: SitePhone) -> PhoneHealth:
    state = adb.shell(phone.serial, _HEALTH_COMMAND, timeout_s=30, check=False)
    processes = adb.shell(phone.serial, _PROCESS_COMMAND, root=phone.root, timeout_s=30, check=False)
    return parse_phone_health(state, processes)


_CAPS_COMMAND = ("for i in {samples}; do for p in /sys/devices/system/cpu/cpufreq/policy*; do "
                 "echo \"S43_CAP ${{p##*/}} $(cat $p/scaling_max_freq 2>/dev/null)\"; done; sleep {period}; done")


def parse_caps_floor(text: str) -> dict[str, int]:
    floor: dict[str, int] = {}
    for policy, cap in re.findall(r"^S43_CAP (\S+) ([0-9]+)$", text, re.M):
        floor[policy] = min(int(cap), floor.get(policy, int(cap)))
    return floor


def read_phone_caps_floor(adb: AdbGuard, phone: SitePhone, *, samples: int = 6, period_s: float = 1.0) -> dict[str, int]:
    """Per CPU cluster, the lowest clock cap seen over a few seconds.

    Some phones move the cap of an idle cluster from one second to the next while awake; the floor is
    what the phone normally runs with. It is recorded when a phone is onboarded and is the bar for
    later runs (a sleeping phone caps every cluster well below it).
    """
    command = _CAPS_COMMAND.format(samples=" ".join(str(index) for index in range(samples)), period=f"{period_s:g}")
    return parse_caps_floor(adb.shell(phone.serial, command, timeout_s=samples * period_s + 30, check=False))


_FACTS_COMMAND = (
    "echo \"S43_MODEL $(getprop ro.product.model)\"; echo \"S43_SOC $(getprop ro.soc.model)\"; "
    "echo \"S43_PLATFORM $(getprop ro.board.platform)\"; echo \"S43_KERNEL $(uname -r)\"; "
    "grep -E '^MemTotal:' /proc/meminfo"
)


def parse_phone_facts(text: str, root_reply: str) -> dict[str, object]:
    soc = _field(text, r"^S43_SOC (\S.*)$") or _field(text, r"^S43_PLATFORM (\S.*)$")
    total = _field(text, r"^MemTotal:\s+([0-9]+) kB")
    facts = {"model": _field(text, r"^S43_MODEL (\S.*)$"), "soc": soc,
             "kernel_release": _field(text, r"^S43_KERNEL (\S+)$"),
             "memory_total_bytes": None if total is None else int(total) * 1024,
             "root": "uid=0" in root_reply}
    missing = sorted(name for name, value in facts.items() if value is None)
    if missing:
        raise PhysicalAdapterError("phone did not report " + ",".join(missing))
    return facts


def read_phone_facts(adb: AdbGuard, serial: str) -> dict[str, object]:
    """Model, SoC, kernel, memory and whether ``su`` answers (the only root command discovery runs)."""
    text = adb.shell(serial, _FACTS_COMMAND, timeout_s=30)
    root_reply = adb.shell(serial, "su -c id 2>/dev/null || true", timeout_s=15, check=False)
    return parse_phone_facts(text, root_reply)
