#!/usr/bin/env python3
"""Calibrate a bounded, energy-unqualified wavefront mechanics profile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PROFILE_SCHEMA = "s42-fp16-burstgpt-gpu-wavefront-profile-v1"
GATE_SCHEMA = "s42-fp16-burstgpt-gpu-wavefront-gate-v1"


class CalibrationError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read {path}") from exc
    if type(value) is not dict:
        raise CalibrationError(f"invalid JSON object: {path}")
    return value


def bridge_result(path: Path) -> dict[str, object]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("FFNDMABUF "):
            rows.append(json.loads(line.removeprefix("FFNDMABUF ")))
    if len(rows) != 1:
        raise CalibrationError("bridge log has no unique FFNDMABUF result")
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path, action="append", required=True)
    parser.add_argument("--bridge-log", type=Path, action="append", required=True)
    parser.add_argument("--window-margin-us", type=int, default=5000)
    parser.add_argument("--service-margin-us", type=int, default=2000)
    parser.add_argument("--service-upper-ppm", type=int, default=1_500_000)
    parser.add_argument("--minimum-decode-windows", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.gate_result) != len(args.bridge_log):
        parser.error("gate results and bridge logs must be paired")
    if len(args.gate_result) < 1:
        parser.error("at least one control run is required")
    if (
        args.window_margin_us < 0
        or args.service_margin_us < 0
        or args.service_upper_ppm < 1_000_000
        or args.minimum_decode_windows <= 0
    ):
        parser.error("invalid calibration margins")
    if not args.output.is_absolute() or args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be an unused absolute path")

    profile = load_json(args.profile)
    if profile.get("schema") != PROFILE_SCHEMA:
        raise CalibrationError("profile schema mismatch")
    durations: list[int] = []
    decode_window_lowers: list[int] = []
    total_decode_windows = 0
    evidence: list[str] = []
    for gate_path, bridge_path in zip(
        args.gate_result, args.bridge_log, strict=True
    ):
        gate = load_json(gate_path)
        bridge = bridge_result(bridge_path)
        if (
            gate.get("schema") != GATE_SCHEMA
            or gate.get("status") != "PASS"
            or gate.get("mode") != "control"
            or gate.get("backfills") != 0
            or gate.get("paid_tail_ns") is None
        ):
            raise CalibrationError("gate result is not a completed control")
        events = gate.get("events")
        if type(events) is not list:
            raise CalibrationError("gate events are absent")
        durations.extend(
            event["duration_us"]
            for event in events
            if type(event) is dict
            and event.get("event") == "worker_execute"
            and event.get("route") == "tail"
            and type(event.get("duration_us")) is int
            and event["duration_us"] > 0
        )
        decode_calls = bridge.get("decode_calls")
        decode_min_ms = bridge.get("prefetch_decode_window_min_ms")
        if (
            bridge.get("status") != "ok"
            or type(decode_calls) is not int
            or decode_calls <= 0
            or type(decode_min_ms) not in {int, float}
            or decode_min_ms <= 0
        ):
            raise CalibrationError("bridge decode-window receipt is invalid")
        total_decode_windows += decode_calls
        decode_window_lowers.append(math.floor(decode_min_ms * 1000))
        evidence.extend((gate_path.name, bridge_path.name))

    if not durations:
        raise CalibrationError("control has no measured tail LM-head calls")
    if total_decode_windows < args.minimum_decode_windows:
        raise CalibrationError("control has too few decode fence windows")
    protected_lower_us = min(decode_window_lowers) - args.window_margin_us
    service_upper_us = (
        max(durations) * args.service_upper_ppm + 999_999
    ) // 1_000_000 + args.service_margin_us
    service_mean_us = sum(durations) // len(durations)
    service_lower_us = min(durations)
    restore_upper_us = 1000
    guard_us = profile["bubble"]["guard_us"]
    if protected_lower_us <= service_upper_us + restore_upper_us + guard_us:
        raise CalibrationError(
            "whole LM-head service does not fit the conservative decode window"
        )

    profile["profile_id"] = "fp16-wavefront-mechanics-calibrated-v1"
    profile["admission"] = "mechanics"
    profile["bubble"] = {
        "guard_us": guard_us,
        "protected_ready_lower_us": protected_lower_us,
        "runtime_verified": True,
    }
    candidate = profile["candidate"]
    candidate["service_latency_us"] = {
        "lower": service_lower_us,
        "mean": service_mean_us,
        "measured": True,
        "sample_count": len(durations),
        "upper": service_upper_us,
    }
    candidate["restore_latency_us"] = {
        "lower": 1,
        "mean": 100,
        "measured": False,
        "sample_count": 0,
        "upper": restore_upper_us,
    }
    profile["evidence_ids"] = sorted(set(
        profile.get("evidence_ids", [])
        + ["physical-control-window-and-tail-calibration", *evidence]
    ))
    profile["calibration"] = {
        "decode_window_lower_samples_us": decode_window_lowers,
        "energy_qualified": False,
        "service_samples": len(durations),
        "service_upper_ppm": args.service_upper_ppm,
        "total_decode_windows": total_decode_windows,
        "window_margin_us": args.window_margin_us,
    }
    args.output.write_text(
        json.dumps(profile, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
