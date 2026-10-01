#!/usr/bin/env python3
"""Fleet energy accountant: desktop + both phones over the paid trace window of one or more runs (read-only).

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.fleet_energy \
        --run legacy=inputs-desktop-legacy-ev2 --run s2a=inputs-two-phone-s2a [--baseline legacy] \
        [--wall s2a=wall.csv] [--charge-efficiency 0.85] [--json OUT.json] [--md OUT.md]

A run directory is an arm's inputs directory (``<inputs>/run-eval/run/RESULT.json`` next to the phone power
logs ``*-POWER.json`` + ``*-POWER-<phone>.txt`` written by run_chain_eval.py, or the ``METER.json`` written by
tools/rig/meter_phones.sh) or any directory holding ``RESULT.json`` and those logs directly;
``--meter label=METER.json`` points at a meter_phones.sh output kept elsewhere (it wins over the run's own logs).

Conventions (every number carries its evidence class: MEASURED, MODELED, or ABSENT):

Desktop (MEASURED): CPU package (RAPL ``package-0``) + GPU board (NVML board power, trapezoid), both over the
  paid window [paid_start_ns, paid_end_ns], taken from RESULT ``trace_energy.fleet_energy_uj_by_domain``. When the
  run directory holds ``resource-samples.jsonl[.gz]`` the same window is re-integrated from the raw samples as a
  cross-check. Not in the boundary (not measured): DRAM, chipset/board, storage, fans, PSU conversion loss, and
  the 5 V the desktop's USB ports deliver to the phones (that energy is counted once, on the phone side).
  Optional wall column (``--wall label=CSV``, header ``host_monotonic_s,watts`` or ``host_epoch_s,watts``):
  integrated over the same window; a wall meter at the desktop plug ALREADY contains the phones' USB input, so
  the wall-based fleet is wall + phones' battery term only (never + USB again).

Phone (MEASURED at the phone's terminals): phone energy = USB input energy + net battery discharge energy.
  USB input  = integral of usb/voltage_now x usb/current_now (1 Hz sysfs samples, trapezoid, window-clipped).
  Battery    = net energy that left the battery over the window = - sum(delta charge_counter x battery
               voltage) (coulomb counter, hardware-integrated; quantization +-2 counter steps). The integral of
               battery/current_now x voltage is reported as a cross-check (sign and unit per phone profile,
               verified against the counter; OP15 reads 0 while charging is disabled).
  This one formula covers charging disabled (battery only discharges) AND charging enabled (energy moved
  USB -> battery is counted in the USB term and credited back through the negative battery term) without double
  counting. With charging enabled the result still contains the charger's conversion loss on the energy stored
  during the window (flag CHARGING_IN_WINDOW): an upper bound of what the workload used.
  Port-equivalent (MODELED, ``--charge-efficiency`` eta): USB + battery / eta, i.e. the USB energy the phone
  would have drawn had its battery state been held constant; eta is an assumption, not a measurement.
  RESULT's ``phone-system`` / ``pixel10pro-phone-system`` domains are the scheduler's ASSUMED 4.5 W / 0.875 W
  model: reported as MODELED for comparison, never added to measured energy.

Fleet (MEASURED boundary) = desktop CPU package + GPU board + both phones' terminal energy. Comparisons with
``--baseline``: host-only saving; conservative fleet saving (treatment desktop + phones vs baseline desktop
only, i.e. phones charged in full and absent from the baseline); attached fleet saving (phones measured in
both arms, idle in the baseline); and the shift ratio = extra phone energy / host energy saved.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "ws3-fleet-energy-v1"
MEASURED, MODELED, ABSENT = "MEASURED", "MODELED", "ABSENT"
UAH_UV_TO_J = 3600e-12  # 1 uAh x 1 uV = 1e-12 Ah*V = 3.6e-9 J
EDGE_TOLERANCE_S = 5.0
COUNTER_JUMP_UAH_PER_S = 5000  # 5 mAh/s = 18 A: no real charge or discharge flow; larger steps are counter resets
GAP_FLAG_S = 5.0

# Battery current conventions, checked against the coulomb counter on 21 longtail_eval_v2 runs (2026-09-30):
#   OP15 (oplus): battery/current_now in mA, positive = discharge (reads 0 at rest with mmi_charging_enable=0).
#     current_now x voltage_now integrates to 0.45-0.53x the counter energy in every run (charging or not), which
#     fits a two-cell series pack reported per cell; the counter agrees with the SoC drop (pe1: 80 -> 75 %,
#     352 mAh of a ~6.8 Ah full counter, 5.2 kJ = 5 % of 7,300 mAh x 3.87 V) -> the counter is authoritative.
#   Pixel 10 Pro (max77779 fuel gauge): battery/current_now in uA, positive = charging; ratio 0.95-1.03.
_OP15 = {"battery_current_to_a": 1e-3, "positive_current_is": "discharge", "current_to_counter_ratio": 0.5}
_PIXEL = {"battery_current_to_a": 1e-6, "positive_current_is": "charge", "current_to_counter_ratio": 1.0}
PHONE_PROFILES: dict[str, dict[str, Any]] = {"op15": _OP15, "3C15AU002CL00000": _OP15,
                                             "pixel": _PIXEL, "5A040DLCH004ES": _PIXEL}
RATIO_TOLERANCE = 0.3  # |current/counter - expected| / expected beyond this flags BATTERY_CROSS_CHECK_RATIO


# ---------------------------------------------------------------- inputs

def find_result(directory: Path) -> Path:
    for candidate in (directory / "run-eval/run/RESULT.json", directory / "RESULT.json", directory / "run/RESULT.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("no RESULT.json under " + str(directory))


def find_meter(directory: Path) -> Path | None:
    """The phone power anchor file: METER.json (meter_phones.sh) else the run's *-run-POWER.json."""
    if (directory / "METER.json").is_file():
        return directory / "METER.json"
    candidates = sorted(directory.glob("*-run-POWER.json")) or sorted(directory.glob("*-POWER.json"))
    return candidates[0] if candidates else None


def parse_sampler(text: str) -> list[dict[str, float]]:
    """Rows of ``uptime_s node=value ...`` (power_sampler.sh / meter_phones.sh); comment lines skipped."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        try:
            row: dict[str, float] = {"uptime_s": float(fields[0])}
        except ValueError:
            continue
        for field in fields[1:]:
            key, _, value = field.partition("=")
            try:
                row[key] = float(int(value))
            except ValueError:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
        rows.append(row)
    return rows


def load_samples(directory: Path) -> list[dict[str, Any]] | None:
    """Host resource samples (plain or gzip JSONL) with GPU and RAPL fields, or None when absent."""
    for name in ("run-eval/run/resource-samples.jsonl", "resource-samples.jsonl", "resource-samples.jsonl.gz",
                 "run-eval/run/resource-samples.jsonl.gz"):
        path = directory / name
        if path.is_file():
            opener = gzip.open if path.suffix == ".gz" else open
            rows = []
            with opener(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(row.get("gpu"), dict) and isinstance(row.get("rapl_package"), dict):
                            rows.append(row)
            return rows
    return None


# ---------------------------------------------------------------- integration helpers

def interpolate(points: Sequence[tuple[float, float]], at: float) -> float:
    """Linear interpolation on sorted (t, y); clamped to the end values outside the covered range."""
    if at <= points[0][0]:
        return points[0][1]
    if at >= points[-1][0]:
        return points[-1][1]
    lo, hi = 0, len(points) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if points[mid][0] <= at:
            lo = mid
        else:
            hi = mid
    (t0, y0), (t1, y1) = points[lo], points[hi]
    return y0 if t1 == t0 else y0 + (y1 - y0) * (at - t0) / (t1 - t0)


def trapezoid(points: Sequence[tuple[float, float]], start: float, end: float) -> float:
    """Integral of the piecewise-linear signal over [start, end] (edge values interpolated/clamped)."""
    if not points or end <= start:
        return 0.0
    bounded = [(start, interpolate(points, start))]
    bounded.extend(point for point in points if start < point[0] < end)
    bounded.append((end, interpolate(points, end)))
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(bounded, bounded[1:]))


def coverage(times: Sequence[float], start: float, end: float) -> dict[str, Any]:
    inside = [t for t in times if start <= t <= end]
    gaps = [b - a for a, b in zip(inside, inside[1:])]
    head = (inside[0] - start) if inside else end - start
    tail = (end - inside[-1]) if inside else end - start
    return {"samples_in_window": len(inside), "head_gap_s": round(head, 3), "tail_gap_s": round(tail, 3),
            "max_gap_s": round(max(gaps + [head, tail]), 3),
            "complete": bool(inside) and head <= EDGE_TOLERANCE_S and tail <= EDGE_TOLERANCE_S}


# ---------------------------------------------------------------- desktop

def desktop_energy(result: Mapping[str, Any], samples: Sequence[Mapping[str, Any]] | None,
                   wall: Sequence[tuple[float, float]] | None = None) -> dict[str, Any]:
    domains = (result.get("trace_energy") or {}).get("fleet_energy_uj_by_domain") or {}
    evidence = (result.get("trace_energy") or {}).get("measurement_evidence_ids") or []
    cpu = domains.get("cpu-package")
    gpu = domains.get("gpu-board")
    out: dict[str, Any] = {
        "cpu_package_j": None if cpu is None else cpu / 1e6, "gpu_board_j": None if gpu is None else gpu / 1e6,
        "evidence": MEASURED if cpu is not None and gpu is not None else ABSENT,
        "measurement_evidence_ids": [item for item in evidence if "rapl" in item or "nvml" in item],
        "not_measured": ["DRAM", "chipset/board", "storage", "fans", "PSU conversion loss",
                         "USB 5 V supplied to the phones (counted on the phone side)"],
    }
    out["host_j"] = (out["cpu_package_j"] + out["gpu_board_j"]) if out["evidence"] == MEASURED else None
    start_ns, end_ns = result["paid_start_ns"], result["paid_end_ns"]
    if samples:
        gpu_points = sorted((int(r["gpu"]["sample_t_ns"]) / 1e9, float(r["gpu"]["power_mw"]) / 1000) for r in samples)
        gpu_j = trapezoid(gpu_points, start_ns / 1e9, end_ns / 1e9)
        rapl_rows = sorted((int(r["rapl_package"]["sample_t_ns"]), int(r["rapl_package"]["energy_uj"]),
                            int(r["rapl_package"]["max_energy_range_uj"])) for r in samples)
        total, prior, unwrapped = 0, rapl_rows[0][1], [(rapl_rows[0][0] / 1e9, 0.0)]
        for sample_ns, energy_uj, maximum in rapl_rows[1:]:
            delta = energy_uj - prior
            if delta < 0:
                delta += maximum
            total += delta
            prior = energy_uj
            unwrapped.append((sample_ns / 1e9, float(total)))
        cpu_j = (interpolate(unwrapped, end_ns / 1e9) - interpolate(unwrapped, start_ns / 1e9)) / 1e6
        out["cross_check"] = {"cpu_package_j": cpu_j, "gpu_board_j": gpu_j, "samples": len(samples),
                              "coverage": coverage([p[0] for p in gpu_points], start_ns / 1e9, end_ns / 1e9)}
        if out["host_j"] is not None:
            out["cross_check"]["host_delta_j"] = cpu_j + gpu_j - out["host_j"]
    if wall:
        out["wall_j"] = trapezoid(wall, start_ns / 1e9, end_ns / 1e9)
        out["wall_evidence"] = MEASURED
    else:
        out["wall_j"], out["wall_evidence"] = None, ABSENT
    return out


def load_wall(path: Path, epoch_minus_monotonic_s: float | None) -> list[tuple[float, float]]:
    """``host_monotonic_s,watts`` or ``host_epoch_s,watts`` CSV -> sorted (host monotonic s, W)."""
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if "host_monotonic_s" in row:
                t = float(row["host_monotonic_s"])
            elif "host_epoch_s" in row and epoch_minus_monotonic_s is not None:
                t = float(row["host_epoch_s"]) - epoch_minus_monotonic_s
            else:
                raise ValueError("wall log needs host_monotonic_s (or host_epoch_s plus a clock anchor)")
            rows.append((t, float(row["watts"])))
    return sorted(rows)


# ---------------------------------------------------------------- phones

def phone_clock(anchor: Mapping[str, Any]) -> tuple[float, float, float]:
    """(host_mono_at_anchor, phone_uptime_at_anchor, rate) mapping phone uptime -> host CLOCK_MONOTONIC s."""
    if "anchor_host_monotonic_s" not in anchor:
        raise ValueError("phone anchor lacks anchor_host_monotonic_s")
    rate = 1.0
    if all(key in anchor for key in ("end_host_epoch_s", "end_phone_uptime_s", "anchor_host_epoch_s")):
        phone_span = anchor["end_phone_uptime_s"] - anchor["anchor_phone_uptime_s"]
        host_span = anchor["end_host_epoch_s"] - anchor["anchor_host_epoch_s"]
        if phone_span > 60 and abs(host_span / phone_span - 1) < 1e-3:
            rate = host_span / phone_span
    return anchor["anchor_host_monotonic_s"], anchor["anchor_phone_uptime_s"], rate


def resolve_profile(label: str, anchor: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    for key in (label, anchor.get("serial", "")):
        if key in PHONE_PROFILES:
            return PHONE_PROFILES[key], "profile:" + key
    return None, "none"


def battery_by_counter(rows: Sequence[Mapping[str, float]], start: float, end: float) -> dict[str, Any] | None:
    """Net battery discharge energy over [start, end] from the coulomb counter: -sum(dq x mean V)."""
    series = [(r["t"], r["battery/charge_counter"], r.get("battery/voltage_now")) for r in rows
              if "battery/charge_counter" in r]
    series = [s for s in series if s[2]]
    if len(series) < 2:
        return None
    first = max([i for i, s in enumerate(series) if s[0] <= start] or [0])
    last = min([i for i, s in enumerate(series) if s[0] >= end] or [len(series) - 1])
    if last <= first:
        return None
    energy = 0.0
    for a, b in zip(series[first:last], series[first + 1:last + 1]):
        energy -= (b[1] - a[1]) * (a[2] + b[2]) / 2 * UAH_UV_TO_J
    steps = [abs(b[1] - a[1]) for a, b in zip(series, series[1:]) if b[1] != a[1]]
    step_uah = min(steps) if steps else None
    # a counter jump much larger than a sampling interval allows (reset, recalibration, reboot) breaks the method
    jumps = [abs(b[1] - a[1]) for a, b in zip(series[first:last], series[first + 1:last + 1])
             if abs(b[1] - a[1]) > COUNTER_JUMP_UAH_PER_S * max(1.0, b[0] - a[0])]
    volts = statistics.fmean(s[2] for s in series[first:last + 1])
    resolution = 2 * step_uah * volts * UAH_UV_TO_J if step_uah else None
    return {"j": energy, "delta_uah": series[last][1] - series[first][1], "step_uah": step_uah,
            "uncertainty_j": resolution, "edge_times": [series[first][0], series[last][0]], "jumps_uah": jumps}


def phone_energy(label: str, anchor: Mapping[str, Any], rows: Sequence[Mapping[str, float]],
                 start: float, end: float, charge_efficiency: float) -> dict[str, Any]:
    """Terminal energy of one phone over the host-monotonic window [start, end]."""
    host0, phone0, rate = phone_clock(anchor)
    mapped = [dict(row, t=host0 + (row["uptime_s"] - phone0) * rate) for row in rows]
    profile, profile_source = resolve_profile(label, anchor)
    flags: list[str] = []
    out: dict[str, Any] = {"serial": anchor.get("serial"), "clock_rate": rate,
                           "anchor_uncertainty_s": anchor.get("anchor_uncertainty_s"),
                           "coverage": coverage([r["t"] for r in mapped], start, end), "profile_source": profile_source}
    if not out["coverage"]["complete"]:
        flags.append("WINDOW_NOT_COVERED")
    usb_points = [(r["t"], r["usb/voltage_now"] * r["usb/current_now"] * 1e-12) for r in mapped
                  if "usb/voltage_now" in r and "usb/current_now" in r]
    out["usb_in_j"] = trapezoid(usb_points, start, end) if usb_points else None
    inside = [r for r in mapped if start <= r["t"] <= end]
    if inside and "usb/input_current_limit" in inside[0]:
        limit = inside[0]["usb/input_current_limit"]
        out["usb_input_limit_ma"] = limit / 1000
        out["usb_at_limit_share"] = round(sum(1 for r in inside if r.get("usb/current_now", 0) >= 0.95 * limit) / len(inside), 4)
    counter = battery_by_counter(mapped, start, end)
    current_j = None
    if profile and any("battery/current_now" in r for r in mapped):
        sign = 1.0 if profile["positive_current_is"] == "discharge" else -1.0
        points = [(r["t"], sign * r["battery/current_now"] * profile["battery_current_to_a"] * r["battery/voltage_now"] * 1e-6)
                  for r in mapped if "battery/current_now" in r and "battery/voltage_now" in r]
        current_j = trapezoid(points, start, end) if points else None
        if inside and all(r.get("battery/current_now", 0) == 0 for r in inside):
            flags.append("BATTERY_CURRENT_READS_ZERO")
            current_j = None
    out["battery_current_cross_check_j"] = current_j
    if counter is not None:
        out["battery_net_discharge_j"] = counter["j"]
        out["battery_method"] = "coulomb_counter"
        out["battery_counter"] = counter
        if counter["jumps_uah"]:
            flags.append("COUNTER_JUMP")
        if current_j is not None and counter["uncertainty_j"] and abs(counter["j"]) > 3 * counter["uncertainty_j"] \
                and abs(current_j) > counter["uncertainty_j"]:
            ratio = current_j / counter["j"]
            out["battery_current_to_counter_ratio"] = ratio
            expected = profile.get("current_to_counter_ratio", 1.0) if profile else 1.0
            if ratio <= 0:
                flags.append("BATTERY_SIGN_MISMATCH")
            elif abs(ratio - expected) / expected > RATIO_TOLERANCE:
                flags.append("BATTERY_CROSS_CHECK_RATIO")
    elif current_j is not None:
        out["battery_net_discharge_j"] = current_j
        out["battery_method"] = "current_now_integral"
        flags.append("BATTERY_FROM_SAMPLED_CURRENT")
    else:
        out["battery_net_discharge_j"] = None
        out["battery_method"] = None
        flags.append("BATTERY_TERM_ABSENT")
    if profile is None:
        flags.append("NO_CURRENT_PROFILE")
    charging = [r for r in inside if "battery/current_now" in r and profile
                and (r["battery/current_now"] > 0) == (profile["positive_current_is"] == "charge")
                and abs(r["battery/current_now"] * profile["battery_current_to_a"]) > 0.02]
    step = (counter or {}).get("step_uah") or 0
    rises = sum(max(0.0, b.get("battery/charge_counter", 0) - a.get("battery/charge_counter", 0))
                for a, b in zip(inside, inside[1:]))
    out["charging_share"] = round(len(charging) / len(inside), 4) if inside else None
    out["battery_only_share"] = round(sum(1 for r in inside if r.get("usb/current_now", 1) < 5000) / len(inside), 4) if inside else None
    out["counter_rise_uah"] = rises
    if (out["charging_share"] or 0) >= 0.02 or (step and rises >= 3 * step):
        flags.append("CHARGING_IN_WINDOW")
    seconds = end - start
    if out["usb_in_j"] is not None and out["battery_net_discharge_j"] is not None:
        out["system_j"] = out["usb_in_j"] + out["battery_net_discharge_j"]
        out["system_w"] = out["system_j"] / seconds
        out["evidence"] = MEASURED
        out["port_equivalent_j"] = out["usb_in_j"] + out["battery_net_discharge_j"] / charge_efficiency
        out["port_equivalent_evidence"] = MODELED
    else:
        out["system_j"] = out["system_w"] = out["port_equivalent_j"] = None
        out["evidence"] = ABSENT
    out["flags"] = sorted(set(flags))
    return out


def modeled_phone(result: Mapping[str, Any]) -> dict[str, float | None]:
    domains = (result.get("trace_energy") or {}).get("fleet_energy_uj_by_domain") or {}
    op15 = domains.get("phone-system")
    pixel = domains.get("pixel10pro-phone-system")
    return {"op15_j": None if op15 is None else op15 / 1e6, "pixel_j": None if pixel is None else pixel / 1e6,
            "evidence": MODELED, "model": ((result.get("trace_energy") or {}).get("estimation_metadata") or {}).get(
                "phone_estimation_version")}


# ---------------------------------------------------------------- one run

def account_run(label: str, directory: Path, charge_efficiency: float = 0.85,
                wall_path: Path | None = None, meter_path: Path | None = None) -> dict[str, Any]:
    result = json.loads(find_result(directory).read_text())
    start_s, end_s = result["paid_start_ns"] / 1e9, result["paid_end_ns"] / 1e9
    meter_path = meter_path or find_meter(directory)
    meter = json.loads(meter_path.read_text()) if meter_path else {}
    epoch_offset = None
    for anchor in meter.values():
        if isinstance(anchor, dict) and "anchor_host_monotonic_s" in anchor:
            epoch_offset = anchor["anchor_host_epoch_s"] - anchor["anchor_host_monotonic_s"]
            break
    wall = load_wall(wall_path, epoch_offset) if wall_path else None
    report: dict[str, Any] = {"label": label, "directory": str(directory), "status": result.get("status"),
                              "duration_s": (result["paid_end_ns"] - result["paid_start_ns"]) / 1e9,
                              "desktop": desktop_energy(result, load_samples(directory), wall),
                              "phones": {}, "modeled_phone": modeled_phone(result),
                              "meter_file": meter_path.name if meter_path else None}
    for phone, anchor in sorted(meter.items()):
        if not isinstance(anchor, dict):
            continue
        local = Path(anchor.get("local", ""))
        path = next((candidate for candidate in (local, meter_path.parent / local.name, directory / local.name)
                     if candidate.is_file()), directory / local.name)
        if not path.is_file():
            report["phones"][phone] = {"evidence": ABSENT, "flags": ["POWER_LOG_MISSING"], "system_j": None}
            continue
        try:
            report["phones"][phone] = phone_energy(phone, anchor, parse_sampler(path.read_text()), start_s, end_s,
                                                   charge_efficiency)
        except ValueError as error:  # e.g. an old POWER.json without the host CLOCK_MONOTONIC anchor
            report["phones"][phone] = {"evidence": ABSENT, "flags": ["CLOCK_ANCHOR_INVALID"], "error": str(error),
                                       "system_j": None}
    host = report["desktop"]["host_j"]
    phones = [entry.get("system_j") for entry in report["phones"].values()]
    report["phones_j"] = sum(phones) if phones and all(v is not None for v in phones) else None
    report["phones_evidence"] = MEASURED if report["phones_j"] is not None else ABSENT
    report["fleet_j"] = host + report["phones_j"] if host is not None and report["phones_j"] is not None else None
    ports = [entry.get("port_equivalent_j") for entry in report["phones"].values()]
    report["fleet_port_equivalent_j"] = (host + sum(ports)) if host is not None and ports and all(
        v is not None for v in ports) else None
    if report["desktop"]["wall_j"] is not None and report["phones"]:
        batteries = [entry.get("battery_net_discharge_j") for entry in report["phones"].values()]
        report["fleet_wall_j"] = (report["desktop"]["wall_j"] + sum(batteries)
                                  if all(v is not None for v in batteries) else None)
    else:
        report["fleet_wall_j"] = None
    return report


def compare(runs: Mapping[str, Mapping[str, Any]], baseline: str) -> dict[str, dict[str, Any]]:
    """Savings of every run against the baseline run (fractions; negative = more energy)."""
    base = runs[baseline]
    base_host = base["desktop"]["host_j"]
    out = {}
    for label, run in runs.items():
        host = run["desktop"]["host_j"]
        row: dict[str, Any] = {"host_saving": None, "fleet_saving_conservative": None,
                               "fleet_saving_attached": None, "shift_ratio": None}
        if host is not None and base_host:
            row["host_saving"] = 1 - host / base_host
            if run["fleet_j"] is not None:
                row["fleet_saving_conservative"] = 1 - run["fleet_j"] / base_host
            if run["fleet_j"] is not None and base["fleet_j"]:
                row["fleet_saving_attached"] = 1 - run["fleet_j"] / base["fleet_j"]
            saved = base_host - host
            if run["phones_j"] is not None and base["phones_j"] is not None and saved > 0:
                row["phone_extra_j"] = run["phones_j"] - base["phones_j"]
                row["shift_ratio"] = row["phone_extra_j"] / saved
        out[label] = row
    return out


# ---------------------------------------------------------------- output

def _kj(value: float | None) -> str:
    return "-" if value is None else "%.1f" % (value / 1e3)


def _pct(value: float | None) -> str:
    return "-" if value is None else "%+.1f %%" % (-100 * value)


def markdown(runs: Mapping[str, Mapping[str, Any]], comparison: Mapping[str, Mapping[str, Any]] | None) -> str:
    lines = ["| run | dur s | CPU kJ | GPU kJ | host kJ | OP15 kJ (W) | Pixel kJ (W) | fleet kJ | phones modeled kJ |"
             " host vs base | fleet vs base (conservative) | fleet vs base (attached) | shift | flags |",
             "|" + "---|" * 14]
    for label, run in runs.items():
        desk = run["desktop"]
        cells = []
        for phone in ("op15", "pixel"):
            entry = run["phones"].get(phone) or {}
            cells.append("-" if entry.get("system_j") is None else "%.1f (%.2f)" % (entry["system_j"] / 1e3, entry["system_w"]))
        modeled = run["modeled_phone"]
        modeled_sum = None if modeled["op15_j"] is None else modeled["op15_j"] + (modeled["pixel_j"] or 0)
        row = (comparison or {}).get(label, {})
        flags = sorted({flag for entry in run["phones"].values() for flag in entry.get("flags", [])})
        lines.append("| %s | %.0f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            label, run["duration_s"], _kj(desk["cpu_package_j"]), _kj(desk["gpu_board_j"]), _kj(desk["host_j"]),
            cells[0], cells[1], _kj(run["fleet_j"]), _kj(modeled_sum), _pct(row.get("host_saving")),
            _pct(row.get("fleet_saving_conservative")), _pct(row.get("fleet_saving_attached")),
            "-" if row.get("shift_ratio") is None else "%.3f" % row["shift_ratio"], " ".join(flags) or "-"))
    return "\n".join(lines) + "\n"


def parse_specs(values: Iterable[str]) -> dict[str, str]:
    out = {}
    for value in values:
        label, sep, rest = value.partition("=")
        if not sep or not label or not rest:
            raise SystemExit("expected label=path, got " + value)
        out[label] = rest
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="label=run directory")
    parser.add_argument("--baseline", help="label of the all-desktop reference run")
    parser.add_argument("--wall", action="append", default=[], help="label=wall-meter CSV (optional)")
    parser.add_argument("--meter", action="append", default=[],
                        help="label=METER.json written by meter_phones.sh outside the run directory (optional)")
    parser.add_argument("--charge-efficiency", type=float, default=0.85,
                        help="MODELED charger efficiency for the port-equivalent column (default 0.85)")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.charge_efficiency <= 1:
        raise SystemExit("--charge-efficiency must be in (0, 1]")
    specs, walls, meters = parse_specs(args.run), parse_specs(args.wall), parse_specs(args.meter)
    runs = {label: account_run(label, Path(path), args.charge_efficiency,
                               Path(walls[label]) if label in walls else None,
                               Path(meters[label]) if label in meters else None)
            for label, path in specs.items()}
    if args.baseline and args.baseline not in runs:
        raise SystemExit("baseline is not one of the runs")
    comparison = compare(runs, args.baseline) if args.baseline else None
    report = {"schema": SCHEMA, "charge_efficiency_modeled": args.charge_efficiency, "baseline": args.baseline,
              "runs": runs, "comparison": comparison}
    text = markdown(runs, comparison)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, sort_keys=True, default=_json_default) + "\n")
    if args.md:
        args.md.write_text(text)
    sys.stdout.write(text)
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(repr(value))


if __name__ == "__main__":
    raise SystemExit(main())
