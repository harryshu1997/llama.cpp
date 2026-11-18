"""The run directory of the portable path: file names, writer and loader.

One directory per run. The names are those of the workstation harnesses (2026-11-16/17), so reports read
their recorded runs too and the new engine's output can be compared with them file by file:

    CONFIG.json  SCHEDULE.json  IDENTITY.json  SERVER_COMMAND.json  server.log
    HELPER_PREFLIGHT.json  HELPER_LAUNCH.json  HELPER_STOP.json | HELPER_STOP_ERROR.json
    PHONE_CLOCK.json  phone-power.txt  PHONE_POWER.json (calibration and quality, new runs)
    IDLE.json  PHASE-<n>.json  PHASE-<n>.disturbed-<k>.json  TAIL_IDLE.json  FOREIGN_LOAD.json
    POWER_SAMPLES.json  POWER_DIAGNOSTICS.json  SERVER_FFN_SHAPES.json  RESULT.json | FAILURE.json
    <request id>.raw (token streams)

``RunDirectory`` loads what the reports need: the phases that counted (never the disturbed attempts),
host energy between two host-clock instants, and the phone's power with its quality.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Mapping

from ...adapters.host_runtime import server_energy_summary
from ..._internal.layer_placement_io import parse_shape_line
from ...drivers.meter import (
    PhoneClock,
    PhonePowerCalibration,
    parse_phone_power,
    phone_power_between,
)


CONFIG = "CONFIG.json"
SCHEDULE = "SCHEDULE.json"
IDENTITY = "IDENTITY.json"
SERVER_COMMAND = "SERVER_COMMAND.json"
SERVER_LOG = "server.log"
SERVER_FFN_SHAPES = "SERVER_FFN_SHAPES.json"
HELPER_PREFLIGHT = "HELPER_PREFLIGHT.json"
HELPER_LAUNCH = "HELPER_LAUNCH.json"
HELPER_STOP = "HELPER_STOP.json"
HELPER_STOP_ERROR = "HELPER_STOP_ERROR.json"
PHONE_CLOCK = "PHONE_CLOCK.json"
PHONE_POWER_TEXT = "phone-power.txt"
PHONE_POWER = "PHONE_POWER.json"
IDLE = "IDLE.json"
TAIL_IDLE = "TAIL_IDLE.json"
FOREIGN_LOAD = "FOREIGN_LOAD.json"
HOST_BUSY = "HOST_BUSY.json"
POWER_SAMPLES = "POWER_SAMPLES.json"
POWER_DIAGNOSTICS = "POWER_DIAGNOSTICS.json"
RESULT = "RESULT.json"
FAILURE = "FAILURE.json"
_PHASE = re.compile(r"PHASE-([0-9]+)")
UNMETERED_PHASE_PREFIXES = ("warm", "rewarm")


class RunDirectoryError(ValueError):
    pass


def phase_file(number: int) -> str:
    return f"PHASE-{number}.json"


def disturbed_phase_file(number: int, attempt: int) -> str:
    return f"PHASE-{number}.disturbed-{attempt}.json"


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="ascii")


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


class RunDirectory:
    """A finished run, read-only."""

    def __init__(self, path: Path, calibration: PhonePowerCalibration | None = None) -> None:
        self.path = Path(path)
        if not (self.path / RESULT).is_file():
            raise RunDirectoryError(f"{self.path} has no {RESULT} (the run did not finish)")
        self.result = _load(self.path / RESULT)
        self.config = _load(self.path / CONFIG)
        self.samples = _load(self.path / POWER_SAMPLES)
        if not self.samples:
            raise RunDirectoryError(f"{self.path} has no host power samples")
        self.last_sample_ns = int(self.samples[-1]["t_ns"])
        numbered = sorted((int(match.group(1)), item) for item in self.path.glob("PHASE-*.json")
                          if (match := _PHASE.fullmatch(item.stem)))
        self.phases = [_load(item) for _, item in numbered]
        clock = self.path / PHONE_CLOCK
        self.clock = PhoneClock.from_json(_load(clock)) if clock.is_file() else None
        text = self.path / PHONE_POWER_TEXT
        self.phone_rows = parse_phone_power(text.read_text(encoding="utf-8", errors="replace")) if text.is_file() else []
        power = _load(self.path / PHONE_POWER) if (self.path / PHONE_POWER).is_file() else {}
        self.assumed_active_w = power.get("assumed_active_w")
        self.assumed_idle_w = power.get("assumed_idle_w")
        if calibration is None and power.get("calibration"):
            calibration = PhonePowerCalibration.from_json(power["calibration"])
        self.calibration = calibration

    # -- shape -----------------------------------------------------------------------------------
    @property
    def n_ff(self) -> int:
        return int(self.config["phone"]["columns"])

    @property
    def shared(self) -> bool:
        return self.config.get("helper") is not None

    @property
    def layers_on_phone(self) -> int:
        helper = self.config.get("helper")
        return len(helper["layers"]) if helper else 0

    @property
    def threads(self) -> str | None:
        arguments = list(self.config.get("server_arguments", ()))
        return arguments[arguments.index("-t") + 1] if "-t" in arguments else None

    def metered_phases(self) -> list[Mapping[str, object]]:
        return [phase for phase in self.phases if not str(phase["name"]).startswith(UNMETERED_PHASE_PREFIXES)]

    def shape_rows(self) -> list[dict[str, object]]:
        """The server's ``S41SERVERFFNSHAPE`` summary rows of this run."""
        path = self.path / SERVER_FFN_SHAPES
        if not path.is_file():
            return []
        return [row for row in (parse_shape_line(line) for line in _load(path)) if row is not None]

    # -- energy ----------------------------------------------------------------------------------
    def host(self, start_ns: int, end_ns: int) -> dict[str, float]:
        energy = server_energy_summary(self.samples, int(start_ns), int(end_ns))
        return {"cpu_j": float(energy["cpu_package_energy_j"]), "gpu_j": float(energy["gpu_board_energy_j"])}

    def phone(self, start_ns: int, end_ns: int, *, active: bool = True) -> dict[str, object] | None:
        """Whole-phone power over the window with its quality; None = absent.

        Without samples the stated power of the site is used (``assumed``): the active power while the
        phone serves calls, the idle power otherwise.
        """
        measured = phone_power_between(self.phone_rows, self.clock, self.calibration, int(start_ns), int(end_ns))
        if measured is not None:
            return {**measured, "quality": "measured"}
        assumed = self.assumed_active_w if active else self.assumed_idle_w
        if assumed is not None and self.shared:
            return {"w": float(assumed), "usb_w": None, "bat_w": None, "n": 0, "quality": "assumed"}
        return None

    def phone_quality(self) -> str:
        """Quality of this run's phone energy: measured, assumed or absent (host-only runs: absent)."""
        if not self.shared:
            return "absent"
        if self.phone_rows and self.clock is not None and self.calibration is not None:
            return "measured"
        return "assumed" if self.assumed_active_w is not None else "absent"

    def idle(self, name: str = IDLE) -> dict[str, object]:
        record = _load(self.path / name)
        energy = record["host_energy"]
        return {"cpu_w": float(energy["cpu_package_average_power_w"]),
                "gpu_w": float(energy["gpu_board_average_power_w"]),
                "phone": self.phone(record["started_ns"], record["finished_ns"], active=False)}

    # -- legs (one request per metered phase, or a split_screen run of 2026-11-16) ----------------
    def legs(self) -> list[dict[str, object]]:
        """Per leg: columns, share, ms per decoded token, package / board / phone W, J per token."""
        if self.phases:
            return [self._phase_leg(phase) for phase in self.metered_phases() if len(phase["requests"]) == 1]
        requests = self.result.get("requests")
        if not requests or "decode_host_energy" not in requests[0]:
            raise RunDirectoryError(f"{self.path} holds no legs")
        tokens = int(self.result["output_tokens_each"])
        return [self._leg(int(row["columns"]), float(row["decode_s"]), tokens - 1,
                          float(row["decode_host_energy"]["cpu_package_energy_j"]),
                          float(row["decode_host_energy"]["gpu_board_energy_j"]),
                          row["first_token_ns"], row["finished_ns"], str(row["request_id"]), None)
                for row in requests]

    def _phase_leg(self, phase: Mapping[str, object]) -> dict[str, object]:
        row = phase["requests"][0]
        first, end = int(row["first_token_ns"]), int(row["finished_ns"])
        energy = self.host(first, end)
        return self._leg(int(row["columns"]), (end - first) / 1e9, int(row["output_tokens"]) - 1,
                         energy["cpu_j"], energy["gpu_j"], first, end, str(phase["name"]), row["tokens_sha256"])

    def _leg(self, columns, decode_s, tokens, cpu_j, gpu_j, first_ns, end_ns, name, sha) -> dict[str, object]:
        phone = self.phone(first_ns, end_ns, active=columns > 0)
        leg = {"name": name, "columns": columns, "share": columns / self.n_ff, "tokens": tokens,
               "ms_token": decode_s / tokens * 1e3, "cpu_w": cpu_j / decode_s, "gpu_w": gpu_j / decode_s,
               "host_j_token": (cpu_j + gpu_j) / tokens,
               "phone_w": phone["w"] if phone else None, "phone_usb_w": phone["usb_w"] if phone else None,
               "phone_bat_w": phone["bat_w"] if phone else None,
               "phone_quality": phone["quality"] if phone else "absent", "tokens_sha256": sha}
        leg["total_j_token"] = leg["host_j_token"] + (phone["w"] * decode_s / tokens if phone else 0.0)
        return leg
