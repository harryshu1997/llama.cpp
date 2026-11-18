"""The doctor: every check that has cost a run, with what to do about it.

``host_checks`` and ``phone_checks`` return ``Check`` rows. A status is

* ``ok``,
* ``degraded``: the run can go on, with a stated consequence (assumed phone power, a USB 2.0 link),
* ``blocking``: starting a metered run now would waste it or disturb someone,
* ``skipped``: the check could not or must not run (a metered run holds the site's lock).

The same rows serve as the gate before a run (``blocking`` rows stop it). Nothing here changes the host
or a phone; where a fix needs a person (an adb server, a kernel boot, waking a phone), the row says so.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
from typing import Callable, Sequence

from ..adapters.contracts import PhysicalAdapterError
from ..adapters.phone_helpers import find_usb_device_by_serial, observe_usb_port
from ..configuration.site import SiteManifest, SitePhone
from .adb import AdbGuard, AdbUnavailable, tcp_listening
from .discover import RAPL_ROOT, USB_ROOT, host_gpus
from .gates import gpu_free, host_busy_jiffies
from .phone import (
    HEXAGON_ARCH_HINTS,
    MAXIMUM_BATTERY_DECIC,
    WORKER_NAME,
    PhoneHealth,
    read_phone_health,
)


STATUSES = ("ok", "degraded", "blocking", "skipped")
LOCK_NAME = "portable-run.lock"


@dataclass(frozen=True)
class Check:
    check_id: str
    status: str
    detail: str
    fix: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise PhysicalAdapterError("unknown check status " + str(self.status))

    def to_json(self) -> dict[str, str]:
        return {"check_id": self.check_id, "detail": self.detail, "fix": self.fix, "status": self.status}


def blocking(checks: Sequence[Check]) -> list[Check]:
    return [check for check in checks if check.status == "blocking"]


def format_checks(checks: Sequence[Check]) -> str:
    width = max((len(check.check_id) for check in checks), default=0)
    lines = []
    for check in checks:
        lines.append(f"{check.status.upper():9s} {check.check_id:{width}s}  {check.detail}")
        if check.fix and check.status != "ok":
            lines.append(" " * (width + 12) + "-> " + check.fix)
    return "\n".join(lines)


class RunLock:
    """One metered run per site: an advisory lock in the site's work root.

    ``with RunLock(work_root):`` holds it for a run; ``RunLock.held(work_root)`` tells the doctor (and
    anything CPU-heavy such as shard slicing) to stay away from the phones and the host meanwhile.
    """

    def __init__(self, work_root: Path) -> None:
        self.path = Path(work_root) / LOCK_NAME
        self._handle = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        try:
            fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._handle.close()
            self._handle = None
            raise PhysicalAdapterError("another metered run holds " + str(self.path)) from error
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(str(os.getpid()) + "\n")
        self._handle.flush()
        return self

    def __exit__(self, *_) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None

    @staticmethod
    def held(work_root: Path) -> bool:
        path = Path(work_root) / LOCK_NAME
        if not path.exists():
            return False
        with path.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
            return False


# --------------------------------------------------------------------------------------------------
# host
# --------------------------------------------------------------------------------------------------


def host_checks(site: SiteManifest, *, run: Callable = subprocess.run, rapl_root: Path = RAPL_ROOT,
                listening: Callable[[int], bool] = tcp_listening, foreign_window_s: float = 2.0,
                sleep: Callable[[float], None] | None = None,
                busy: Callable[[], int] = host_busy_jiffies, quiet_cores: float = 4.0) -> list[Check]:
    import time
    checks: list[Check] = []
    host, paths = site.host, site.paths
    # files
    missing = [str(path) for path in (paths.server_path, paths.adb_path, *paths.models.values(),
                                      *paths.host_tools.values()) if not Path(path).exists()]
    checks.append(Check("host-files", "blocking" if missing else "ok",
                        "missing: " + ", ".join(missing) if missing else "server, adb, models and host tools exist",
                        "correct the paths in the site file" if missing else ""))
    # GPU identity and occupancy
    try:
        live = {gpu["index"]: gpu for gpu in host_gpus(run=run)}
        serving = live.get(host.serving_gpu_index)
        if serving is None or serving["uuid"] != host.serving_gpu.uuid:
            checks.append(Check("host-gpu", "blocking", f"GPU {host.serving_gpu_index} is not the board of the site "
                                f"file ({host.serving_gpu.uuid})", "run discover to refresh the site's facts"))
        else:
            checks.append(Check("host-gpu", "ok", f"GPU {serving['index']} {serving['name']} {serving['uuid']}"))
            state = gpu_free(host.serving_gpu_index, run=run)
            checks.append(Check(
                "host-gpu-free", "ok" if state["free"] else "blocking",
                f"{state['memory_used_mib']} MiB in use, compute processes {state['compute_pids'] or 'none'}",
                "" if state["free"] else "another job holds the serving board; wait for it (it is not ours to stop)"))
    except (OSError, subprocess.SubprocessError, ValueError, PhysicalAdapterError) as error:
        checks.append(Check("host-gpu", "blocking", "GPU state is unreadable: " + str(error)[:160],
                            "check nvidia-smi on this host"))
    # RAPL
    unreadable = []
    for zone in host.rapl_zones or ("intel-rapl:0",):
        try:
            int(Path(rapl_root, zone, "energy_uj").read_text(encoding="ascii"))
        except (OSError, ValueError):
            unreadable.append(zone)
    if not host.rapl_zones or unreadable:
        checks.append(Check("host-rapl", "blocking", "CPU energy counter unreadable: "
                            + ", ".join(unreadable or ["no zone in the site file"]),
                            "an administrator makes energy_uj readable (it resets at every boot); then run discover"))
    else:
        checks.append(Check("host-rapl", "ok", "readable: " + ", ".join(host.rapl_zones)
                            + ("" if len(host.rapl_zones) == 1 else " (only the first package is metered)")))
    # other users' load, right now
    before = busy()
    (sleep or time.sleep)(foreign_window_s)
    cores = (busy() - before) / os.sysconf("SC_CLK_TCK") / foreign_window_s
    checks.append(Check("host-load", "ok" if cores < quiet_cores else "degraded",
                        f"{cores:.1f} cores busy on the host over {foreign_window_s:.0f} s",
                        "" if cores < quiet_cores else "a run waits for a quiet host before every phase and repeats "
                                                       "a disturbed one; nothing of other users is touched"))
    # adb servers
    for port in sorted({phone.adb_port for phone in site.phones}):
        up = listening(port)
        checks.append(Check(f"adb-server:{port}", "ok" if up else "blocking",
                            "a server listens" if up else "no adb server listens on this port",
                            "" if up else "the owner of the phones starts the adb server; none is started here"))
    return checks


# --------------------------------------------------------------------------------------------------
# phones
# --------------------------------------------------------------------------------------------------


def health_checks(phone: SitePhone, health: PhoneHealth, *, metered: bool = True,
                  required_caps_khz: dict[str, int] | None = None,
                  needed_memory_bytes: int = 0) -> list[Check]:
    """Checks on a phone's state. ``required_caps_khz``: the CPU caps recorded when the phone was
    onboarded (a run needs at least those); absent = the hardware maxima."""
    name = phone.device_id
    stop = "blocking" if metered else "degraded"
    checks = []
    if health.workload_processes:
        checks.append(Check(f"phone-idle:{name}", "blocking", "LLM workload running: "
                            + ", ".join(health.workload_processes),
                            "someone is using this phone (or a worker was left behind); do not start a run on it"))
    else:
        checks.append(Check(f"phone-idle:{name}", "ok", "no LLM workload process"))
    if health.awake is False:
        checks.append(Check(f"phone-awake:{name}", stop, "the phone is asleep (its CPU clocks are capped)",
                            "wake the phone and keep it awake while it helps (a phone setting, the owner's)"))
    elif health.awake is None:
        checks.append(Check(f"phone-awake:{name}", "degraded", "wakefulness unknown"))
    else:
        checks.append(Check(f"phone-awake:{name}", "ok", "awake"))
    maxima = dict(health.cpu_maximum_khz)
    wanted = required_caps_khz or maxima
    low = {policy: cap for policy, cap in health.cpu_caps_khz.items() if cap < wanted.get(policy, 0)}
    fastest = max(maxima, key=maxima.get) if maxima else None
    listing = ", ".join(f"{policy} {cap // 1000} MHz (needs {wanted[policy] // 1000})"
                        for policy, cap in sorted(low.items()))
    if low and (required_caps_khz or fastest in low):
        # below the caps recorded at onboarding, or (no record yet) the fastest cluster itself is capped
        checks.append(Check(f"phone-cpu:{name}", stop, "CPU clocks capped: " + listing,
                            "an awake, cool phone lifts the caps; a run with capped clocks is slower than the "
                            "onboarding measured"))
    elif low:
        checks.append(Check(f"phone-cpu:{name}", "degraded", "a slower CPU cluster is capped: " + listing,
                            "normal on some phones while awake; onboard-phone records the caps this phone runs with"))
    elif health.cpu_caps_khz:
        checks.append(Check(f"phone-cpu:{name}", "ok", "CPU clocks " + ", ".join(
            f"{policy} {cap // 1000} MHz" for policy, cap in sorted(health.cpu_caps_khz.items()))))
    hot = health.battery_decic is not None and health.battery_decic >= MAXIMUM_BATTERY_DECIC
    throttled = health.thermal_status is not None and health.thermal_status > 0
    if hot or throttled:
        checks.append(Check(f"phone-thermal:{name}", stop,
                            f"battery {None if health.battery_decic is None else health.battery_decic / 10} C, "
                            f"thermal status {health.thermal_status}", "let the phone cool before a metered run"))
    else:
        checks.append(Check(f"phone-thermal:{name}", "ok",
                            f"battery {None if health.battery_decic is None else health.battery_decic / 10} C, "
                            f"thermal status {health.thermal_status}"))
    if health.charging is False or (health.battery_level is not None and health.battery_level < 20):
        checks.append(Check(f"phone-battery:{name}", "degraded",
                            f"level {health.battery_level} %, charging {health.charging}"))
    else:
        checks.append(Check(f"phone-battery:{name}", "ok", f"level {health.battery_level} %, charging {health.charging}"))
    available = health.memory_available_bytes
    if available is not None and available < needed_memory_bytes:
        checks.append(Check(f"phone-memory:{name}", "blocking",
                            f"{available / 1e9:.2f} GB available, the shards need {needed_memory_bytes / 1e9:.2f} GB",
                            "free memory on the phone or plan a smaller share (onboard-phone)"))
    else:
        checks.append(Check(f"phone-memory:{name}", "ok" if available is not None else "degraded",
                            "available memory unknown" if available is None else f"{available / 1e9:.2f} GB available"))
    return checks


def phone_checks(site: SiteManifest, phone: SitePhone, adb: AdbGuard, *, metered: bool = True,
                 usb_root: Path = USB_ROOT, listening: Callable[[int], bool] = tcp_listening,
                 required_caps_khz: dict[str, int] | None = None, needed_memory_bytes: int = 0,
                 lock_held: Callable[[Path], bool] = RunLock.held) -> list[Check]:
    name = phone.device_id
    if lock_held(site.paths.work_root):
        return [Check(f"phone:{name}", "skipped", "a metered run holds the site lock; the phone is not touched")]
    checks: list[Check] = []
    # host side: the USB link
    device = find_usb_device_by_serial(phone.serial, sysfs_root=usb_root)
    if device is None:
        return [Check(f"phone-usb:{name}", "blocking", "no USB device has serial " + phone.serial,
                      "attach the phone (or it is in its direct USB mode and a session must end first)")]
    port = observe_usb_port(device, sysfs_root=usb_root)
    if port.negotiated_speed_mbps < phone.minimum_usb_speed_mbps:
        checks.append(Check(f"phone-usb:{name}", "blocking", f"{port.sysfs_device} at {port.negotiated_speed_mbps} "
                            f"Mb/s, the site needs {phone.minimum_usb_speed_mbps}",
                            "use a faster port or cable, or lower minimum_usb_speed_mbps in the site file"))
    else:
        moved = "" if port.sysfs_device == phone.usb_sysfs_device else \
            f" (the site file says {phone.usb_sysfs_device}: run discover)"
        slow = port.negotiated_speed_mbps < 5000
        checks.append(Check(f"phone-usb:{name}", "degraded" if slow or moved else "ok",
                            f"{port.sysfs_device} at {port.negotiated_speed_mbps} Mb/s"
                            + (", behind a hub" if port.behind_hub else "") + moved,
                            "a USB 2.0 link adds latency to every call; onboarding measures what is left" if slow else ""))
    # adb
    try:
        state = dict(adb.devices()).get(phone.serial)
    except AdbUnavailable as error:
        return checks + [Check(f"phone-adb:{name}", "blocking", str(error))]
    if state != "device":
        return checks + [Check(f"phone-adb:{name}", "blocking", f"adb state {state or 'absent'}",
                               "authorize this host on the phone (USB debugging), or wait for the phone to return to adb")]
    checks.append(Check(f"phone-adb:{name}", "ok", "authorized"))
    # kernel for the direct USB transport
    if phone.transport == "functionfs-bridge":
        release = adb.shell(phone.serial, "uname -r", timeout_s=15).strip()
        wanted = str(phone.functionfs.get("kernel_release"))
        if release != wanted:
            checks.append(Check(f"phone-kernel:{name}", "blocking", f"kernel {release}, the direct USB link needs {wanted}",
                                "the qualified kernel is RAM-booted by the boot helper kept with the site "
                                "(each boot needs the owner's approval; nothing here runs it), or use transport adb-tcp"))
        else:
            checks.append(Check(f"phone-kernel:{name}", "ok", release))
    if not phone.root:
        checks.append(Check(f"phone-root:{name}", "degraded", "not rooted: no direct USB link, power is assumed unless "
                            "its sysfs nodes are readable"))
    # bundle and backend library
    worker = phone.bundle_directory + "/" + WORKER_NAME
    listing = adb.shell(phone.serial, "ls " + shlex.quote(phone.bundle_directory) + " 2>/dev/null", root=phone.as_root,
                        timeout_s=30, check=False).split()
    arch = HEXAGON_ARCH_HINTS.get(phone.soc)
    library = f"libggml-htp-{arch}.so" if arch and phone.kind == "htp" else None
    if WORKER_NAME not in listing:
        checks.append(Check(f"phone-bundle:{name}", "blocking", worker + " is absent",
                            "push the worker bundle built for this phone (provision --bundle DIR)"))
    elif library and library not in listing:
        checks.append(Check(f"phone-bundle:{name}", "blocking", f"{phone.soc} needs {library}, the bundle has "
                            + (", ".join(sorted(x for x in listing if x.startswith("libggml-htp"))) or "no NPU library"),
                            "build the NPU library for this SoC and push it with the bundle"))
    else:
        checks.append(Check(f"phone-bundle:{name}", "ok", f"{len(listing)} files" + (f", {library}" if library else "")))
    # power
    power = phone.power
    if power is None:
        checks.append(Check(f"phone-power:{name}", "degraded", "no power source declared: phone energy will be absent",
                            "declare the sysfs nodes or assumed powers in the site file"))
    elif not power.measured:
        checks.append(Check(f"phone-power:{name}", "degraded",
                            f"assumed {power.assumed_active_w} W active / {power.assumed_idle_w} W idle"))
    else:
        nodes = [power.usb_current_node, power.usb_voltage_node, power.battery_current_node,
                 power.battery_voltage_node]
        paths = [node if node.startswith("/") else "/sys/class/power_supply/" + node for node in nodes]
        readable = adb.shell(phone.serial, "for f in " + " ".join(shlex.quote(p) for p in paths)
                             + "; do test -r $f && echo ok; done", root=power.needs_root, timeout_s=15,
                             check=False).split().count("ok")
        if readable != len(paths):
            checks.append(Check(f"phone-power:{name}", "blocking", f"{len(paths) - readable} power nodes unreadable",
                                "correct the nodes, set needs_root, or declare assumed powers in the site file"))
        else:
            checks.append(Check(f"phone-power:{name}", "ok" if power.gauge_ratio_provenance != "uncalibrated"
                                else "degraded", f"measured, gauge ratio {power.gauge_ratio} "
                                f"({power.gauge_ratio_provenance})"))
    # host ports this phone's session will bind
    ports = [phone.forward_port] + ([int(phone.functionfs["bridge_port"])] if phone.functionfs
                                    and "bridge_port" in phone.functionfs else [])
    busy_ports = [port_number for port_number in ports if listening(port_number)]
    checks.append(Check(f"phone-ports:{name}", "blocking" if busy_ports else "ok",
                        "in use: " + ", ".join(map(str, busy_ports)) if busy_ports else "free: " + ", ".join(map(str, ports)),
                        "a session of ours is still up or another program uses the port" if busy_ports else ""))
    return checks + health_checks(phone, read_phone_health(adb, phone), metered=metered,
                                  required_caps_khz=required_caps_khz, needed_memory_bytes=needed_memory_bytes)


def site_checks(site: SiteManifest, *, phones: Sequence[str] | None = None, metered: bool = True,
                required_caps_by_phone: dict[str, dict[str, int]] | None = None, **options) -> list[Check]:
    """Host checks and the checks of the named phones (default: every phone of the site).

    ``required_caps_by_phone``: per phone, the CPU clock caps it had when it was onboarded (from the site
    profile). A phone that always runs with a capped cluster is measured that way and held to that; a
    phone without a record is held to its hardware maxima.
    """
    checks = host_checks(site)
    for phone in site.phones:
        if phones is not None and phone.device_id not in phones:
            continue
        caps = (required_caps_by_phone or {}).get(phone.device_id) or None
        checks += phone_checks(site, phone, AdbGuard(site.paths.adb_path, phone.adb_port), metered=metered,
                               required_caps_khz=caps, **options)
    return checks
