"""Energy meters of the portable path: host package + GPU board, and a phone's whole power.

Every energy window carries a quality: ``measured`` (sampled inside the window), ``assumed`` (a stated
power times the window) or ``absent`` (nothing usable), and a report repeats it next to the number.

Phone power is sampled on the phone by a sysfs sampler that writes one line per second::

    <uptime_s> usb_ua=<uA> usb_uv=<uV> bat_ma=<gauge current> bat_uv=<uV> ...

The phone's clock is paired with the host's monotonic clock once per run (``PhoneClock``). Whole-phone
power = USB input + battery; the battery term is the gauge current scaled by a per-phone calibration
(:class:`PhonePowerCalibration`: unit, sign and the gauge ratio), which belongs to the site description
and never to code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex
import statistics
import time
from typing import Callable, Mapping, Sequence

from ..adapters.contracts import PhysicalAdapterError
from ..adapters.host_runtime import HostEnergySampler, default_host_metric_callbacks, server_energy_summary


QUALITIES = ("measured", "assumed", "absent")
BATTERY_CURRENT_UNITS = ("mA", "uA")
MINIMUM_PHONE_SAMPLES = 3
_NUMBER = re.compile(r"-?[0-9.]+")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalAdapterError(message)


@dataclass(frozen=True)
class PhonePowerCalibration:
    """How a phone's fuel-gauge current becomes watts.

    ``battery_discharge_sign`` +1: a positive reading means the battery discharges; -1: positive means it
    charges. ``gauge_ratio``: gauge reading / true current (1.0 with provenance ``uncalibrated`` until a
    ratio was measured against the USB input).
    """

    battery_current_unit: str = "mA"
    battery_discharge_sign: int = 1
    gauge_ratio: float = 1.0
    gauge_ratio_provenance: str = "uncalibrated"

    def __post_init__(self) -> None:
        _require(self.battery_current_unit in BATTERY_CURRENT_UNITS, "battery current unit must be mA or uA")
        _require(self.battery_discharge_sign in (1, -1), "battery discharge sign must be +1 or -1")
        _require(isinstance(self.gauge_ratio, (int, float)) and self.gauge_ratio > 0.0, "gauge ratio must be positive")
        _require(bool(self.gauge_ratio_provenance), "gauge ratio provenance is empty")

    def battery_w(self, current: float, microvolts: float) -> float:
        amperes = current * (1e-3 if self.battery_current_unit == "mA" else 1e-6) * self.battery_discharge_sign
        return amperes * microvolts * 1e-6 / self.gauge_ratio

    def to_json(self) -> dict[str, object]:
        return {"battery_current_unit": self.battery_current_unit,
                "battery_discharge_sign": self.battery_discharge_sign,
                "gauge_ratio": self.gauge_ratio, "gauge_ratio_provenance": self.gauge_ratio_provenance}

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "PhonePowerCalibration":
        return cls(str(value["battery_current_unit"]), int(value["battery_discharge_sign"]),
                   float(value["gauge_ratio"]), str(value["gauge_ratio_provenance"]))


@dataclass(frozen=True)
class PhoneClock:
    """One pairing of the phone's uptime with the host's monotonic clock."""

    phone_uptime_s: float
    host_monotonic_ns: int

    def uptime_s(self, host_monotonic_ns: int) -> float:
        return self.phone_uptime_s + (host_monotonic_ns - self.host_monotonic_ns) / 1e9

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "PhoneClock":
        return cls(float(value["phone_uptime_s"]), int(value["host_monotonic_ns"]))


def parse_phone_power(text: str) -> list[dict[str, float]]:
    """Sampler lines -> rows ``{"t": uptime_s, <field>: value, ...}`` (comments and broken lines dropped)."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or line.startswith("#"):
            continue
        try:
            row = {key: float(value) for key, value in (part.split("=", 1) for part in parts[1:])
                   if _NUMBER.fullmatch(value)}
            row["t"] = float(parts[0])
        except ValueError:
            continue
        rows.append(row)
    return rows


def phone_power_between(rows: Sequence[Mapping[str, float]], clock: PhoneClock | None,
                        calibration: PhonePowerCalibration | None, start_ns: int,
                        end_ns: int) -> dict[str, object] | None:
    """Mean whole-phone power over a host-clock window, or None when it cannot be measured.

    None = no clock pairing, no calibration, no USB fields, or fewer than three samples inside.
    """
    if clock is None or calibration is None or not rows:
        return None
    first, last = clock.uptime_s(start_ns), clock.uptime_s(end_ns)
    inside = [row for row in rows if first <= row["t"] <= last]
    if len(inside) < MINIMUM_PHONE_SAMPLES or any(
            name not in inside[0] for name in ("usb_ua", "usb_uv", "bat_ma", "bat_uv")):
        return None
    usb = statistics.fmean(row["usb_ua"] * row["usb_uv"] * 1e-12 for row in inside)
    battery = statistics.fmean(calibration.battery_w(row["bat_ma"], row["bat_uv"]) for row in inside)
    def celsius(row: Mapping[str, float]) -> float:
        # a thermal zone in milli-C (older samplers) or the battery node in deci-C
        return row["battery"] / 1000.0 if "battery" in row else row.get("bat_dc", 0.0) / 10.0

    return {"w": usb + battery, "usb_w": usb, "bat_w": battery, "n": len(inside),
            "battery_c": (celsius(inside[0]), max(celsius(row) for row in inside), celsius(inside[-1]))}


# --------------------------------------------------------------------------------------------------
# host meter
# --------------------------------------------------------------------------------------------------


class HostMeter:
    """RAPL package + the board of one GPU (by index), sampled in a child process."""

    def __init__(self, gpu_index: int, *, interval_s: float = 0.1, sampler_factory: Callable = HostEnergySampler,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.gpu_index = gpu_index
        self._sampler = sampler_factory(default_host_metric_callbacks(gpu_index=gpu_index), interval_s=interval_s)
        self._clock, self._sleep = clock, sleep

    def start(self) -> None:
        self._sampler.start()

    def stop(self) -> None:
        self._sampler.stop()

    def rows(self) -> tuple[dict[str, object], ...]:
        return self._sampler.rows()

    def wait_past(self, instant_ns: int, timeout_s: float = 15.0) -> None:
        """Block until a GPU and a RAPL sample exist after ``instant_ns`` (interpolation needs them)."""
        deadline = self._clock() + timeout_s
        while self._clock() < deadline:
            rows = [row for row in self._sampler.rows()
                    if type(row.get("gpu")) is dict and type(row.get("rapl_package")) is dict]
            if any(int(row["gpu"]["sample_t_ns"]) > instant_ns and int(row["rapl_package"]["sample_t_ns"]) > instant_ns
                   for row in rows[-4:]):
                return
            self._sleep(0.05)
        raise PhysicalAdapterError("host meter has no sample after the window")

    def energy(self, start_ns: int, end_ns: int) -> dict[str, object]:
        """CPU package and GPU board energy over a host-clock window."""
        self.wait_past(end_ns)
        return dict(server_energy_summary(self._sampler.rows_between(start_ns, end_ns), start_ns, end_ns))

    def save(self, directory: Path) -> None:
        for name, value in (("POWER_SAMPLES.json", self._sampler.rows()),
                            ("POWER_DIAGNOSTICS.json", self._sampler.diagnostics())):
            Path(directory, name).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                                             encoding="ascii")


# --------------------------------------------------------------------------------------------------
# phone meters
# --------------------------------------------------------------------------------------------------

SAMPLER_NAME = "phone_power_sampler.sh"
CPU_POLICY_ROOT = "/sys/devices/system/cpu/cpufreq"


class PhoneSysfsMeter:
    """A phone's whole power from its own sysfs nodes, sampled on the phone for the whole session.

    The generic sampler script (``campaigns/burstgpt/tools/rig/phone_power_sampler.sh``, read-only) is
    kept under the site's work directory on the phone and writes there. It keeps running while the phone
    is away from adb (its direct USB mode) and is collected when adb returns. The collected lines are
    stored with the run-directory's field names (``usb_ua usb_uv bat_ma bat_uv bat_dc m<policy>``).
    """

    quality = "measured"

    def __init__(self, adb, phone, sampler_script: Path, *, period_s: float = 1.0, max_samples: int = 86400,
                 cpu_policies: Sequence[str] = (), clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        power = phone.power
        if power is None or not power.measured:
            raise PhysicalAdapterError("phone has no measured power source in the site file")
        self.adb, self.phone, self.sampler_script = adb, phone, Path(sampler_script)
        self.period_s, self.max_samples, self._clock_ns = period_s, max_samples, clock_ns
        self.directory = phone.work_directory + "/sampler"
        self.nodes = {power.usb_current_node: "usb_ua", power.usb_voltage_node: "usb_uv",
                      power.battery_current_node: "bat_ma", power.battery_voltage_node: "bat_uv",
                      "battery/temp": "bat_dc"}
        for policy in cpu_policies:
            self.nodes[f"{CPU_POLICY_ROOT}/{policy}/scaling_max_freq"] = "m" + policy.removeprefix("policy")
        self.calibration = PhonePowerCalibration(power.battery_current_unit, power.battery_discharge_sign,
                                                 power.gauge_ratio, power.gauge_ratio_provenance)
        self._token: str | None = None

    def _shell(self, command: str, *, timeout_s: float = 60, check: bool = True) -> str:
        return self.adb.shell(self.phone.serial, command, root=self.phone.power.needs_root, timeout_s=timeout_s,
                              check=check)

    def start(self, run_directory: Path, token: str) -> dict[str, object]:
        """Start the sampler and pair the clocks; returns the pairing (also saved as PHONE_CLOCK.json)."""
        from .adb import sha256_file
        remote = self.directory + "/" + SAMPLER_NAME
        self.adb.push_verified(self.phone.serial, self.sampler_script, remote, sha256_file(self.sampler_script),
                               allowed_roots=(self.phone.work_directory,), root=self.phone.power.needs_root)
        out, stop = f"{self.directory}/{token}.txt", f"{self.directory}/{token}.stop"
        self._shell("rm -f " + shlex.quote(out) + " " + shlex.quote(stop))
        self._shell("nohup sh " + shlex.quote(remote) + " " + shlex.quote(out) + " " + shlex.quote(stop)
                    + f" {self.period_s:g} {self.max_samples} " + " ".join(shlex.quote(node) for node in self.nodes)
                    + " > /dev/null 2>&1 < /dev/null &", timeout_s=30)
        before = self._clock_ns()
        uptime = float(self.adb.shell(self.phone.serial, "cat /proc/uptime", timeout_s=15).split()[0])
        after = self._clock_ns()
        self._token = token
        pairing = {"phone_uptime_s": uptime, "host_monotonic_ns": (before + after) // 2,
                   "pairing_width_ns": after - before}
        _write_json(Path(run_directory, "PHONE_CLOCK.json"), pairing)
        return pairing

    def collect(self, run_directory: Path) -> int:
        """Stop the sampler, store its lines as ``phone-power.txt`` and the calibration; returns the samples."""
        if self._token is None:
            raise PhysicalAdapterError("phone meter was not started")
        out, stop = f"{self.directory}/{self._token}.txt", f"{self.directory}/{self._token}.stop"
        # the phone may still be returning from its direct USB mode
        self.adb.run(self.phone.serial, "wait-for-device", timeout_s=180, check=False)
        self._shell("touch " + shlex.quote(stop) + "; sleep 2; true", timeout_s=30)
        text = self._shell("cat " + shlex.quote(out), timeout_s=180)
        lines = self.normalise(text)
        Path(run_directory, "phone-power.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
        _write_json(Path(run_directory, "PHONE_POWER.json"), {"quality": self.quality,
                                                              "calibration": self.calibration.to_json()})
        self._shell("rm -f " + shlex.quote(out) + " " + shlex.quote(stop), check=False)
        self._token = None
        return sum(1 for line in lines if line and not line.startswith("#"))

    def normalise(self, text: str) -> list[str]:
        """Sampler lines with their node names replaced by the run-directory's field names."""
        lines = []
        for line in text.splitlines():
            if not line.strip() or line.startswith("#"):
                lines.append(line)
                continue
            parts = line.split()
            fields = []
            for part in parts[1:]:
                node, separator, value = part.partition("=")
                if separator and node in self.nodes:
                    fields.append(self.nodes[node] + "=" + value)
            lines.append(" ".join([parts[0], *fields]))
        return lines


class AssumedPhoneMeter:
    """A phone whose power cannot be read: the site's stated active and idle power, marked ``assumed``."""

    quality = "assumed"

    def __init__(self, phone) -> None:
        power = phone.power
        if power is None or power.assumed_active_w is None:
            raise PhysicalAdapterError("phone has no assumed power in the site file")
        self.active_w, self.idle_w = power.assumed_active_w, power.assumed_idle_w

    def start(self, run_directory: Path, token: str) -> dict[str, object]:
        return {}

    def collect(self, run_directory: Path) -> int:
        _write_json(Path(run_directory, "PHONE_POWER.json"), {"quality": self.quality,
                                                              "assumed_active_w": self.active_w,
                                                              "assumed_idle_w": self.idle_w})
        return 0


def phone_meter(adb, phone, sampler_script: Path, **options):
    """The meter the site declares for this phone: measured, assumed, or None (its energy will be absent)."""
    if phone.power is None:
        return None
    if phone.power.measured:
        return PhoneSysfsMeter(adb, phone, sampler_script, **options)
    return AssumedPhoneMeter(phone)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
