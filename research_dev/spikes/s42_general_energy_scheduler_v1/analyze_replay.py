#!/usr/bin/env python3
"""Compare S42 replay predictions with the imported physical campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from replay import OUTPUT_SCHEMA, ReplayError, canonical_bytes, load_profile


SCHEMA = "s42-general-scheduler-analysis-v1"


class AnalysisError(ValueError):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_replay(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="ascii") as source:
            result = json.load(source, object_pairs_hook=_no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read replay: {exc}") from exc
    if type(result) is not dict or result.get("schema") != OUTPUT_SCHEMA:
        raise AnalysisError("replay schema mismatch")
    supplied = result.get("replay_hash")
    if type(supplied) is not str or not supplied.startswith("sha256:"):
        raise AnalysisError("replay hash is missing")
    unhashed = dict(result)
    del unhashed["replay_hash"]
    expected = "sha256:" + hashlib.sha256(canonical_bytes(unhashed)).hexdigest()
    if supplied != expected:
        raise AnalysisError("replay hash mismatch")
    return result


def relative_error_ppm(predicted: int, observed: int) -> int:
    return int(round(1_000_000 * (predicted - observed) / observed))


def analyze(profile_path: Path, replay_path: Path) -> dict[str, Any]:
    raw_profile, profile = load_profile(profile_path)
    replay = load_replay(replay_path)
    if replay.get("profile", {}).get("profile_hash") != raw_profile.get("profile_hash"):
        raise AnalysisError("replay does not bind the supplied profile")
    raw_modes = replay.get("modes")
    if type(raw_modes) is not list:
        raise AnalysisError("replay modes are missing")
    modes = {
        row.get("mode"): row
        for row in raw_modes
        if type(row) is dict and type(row.get("mode")) is str
    }
    if "control" not in modes or "shadow" not in modes or "enforce" not in modes:
        raise AnalysisError("required replay modes are missing")
    observed = raw_profile.get("calibration", {}).get("observed_campaigns")
    if type(observed) is not dict:
        raise AnalysisError("profile lacks observed campaigns")
    comparisons = []
    for mode, campaign_name in (("control", "control"), ("shadow", "cpu_op15")):
        campaign = observed.get(campaign_name)
        predicted = modes[mode]
        if type(campaign) is not dict:
            raise AnalysisError("observed campaign is invalid")
        observed_makespan = campaign.get("makespan_us")
        predicted_makespan = predicted.get("makespan_us")
        observed_slo = campaign.get("slo_met")
        predicted_slo = predicted.get("conservative_slo_met")
        if any(
            type(value) is not int
            for value in (
                observed_makespan,
                predicted_makespan,
                observed_slo,
                predicted_slo,
            )
        ):
            raise AnalysisError("comparison metric is invalid")
        error_ppm = relative_error_ppm(predicted_makespan, observed_makespan)
        comparisons.append({
            "mode": mode,
            "campaign": campaign_name,
            "observed_makespan_us": observed_makespan,
            "predicted_makespan_us": predicted_makespan,
            "makespan_error_ppm": error_ppm,
            "observed_slo_met": observed_slo,
            "predicted_conservative_slo_met": predicted_slo,
            "pass": abs(error_ppm) <= 20_000 and observed_slo == predicted_slo,
        })
    enforce = modes["enforce"]
    enforce_counts = enforce.get("route_counts", {})
    phone_enforced = (
        enforce_counts.get("cold-cpu-op15-ffn", 0)
        if type(enforce_counts) is dict
        else -1
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "profile_id": profile.profile_id,
        "profile_hash": raw_profile["profile_hash"],
        "replay_hash": replay["replay_hash"],
        "comparisons": comparisons,
        "enforce_phone_request_count": phone_enforced,
        "energy_verdict": (
            "FAIL_CLOSED_NO_PHONE_OFFLOAD_NO_SYNCHRONIZED_FLEET_ENERGY"
            if phone_enforced == 0
            else "INVALID_ENFORCE_OFFLOAD_WITHOUT_ENERGY"
        ),
        "pass": (
            all(item["pass"] for item in comparisons)
            and phone_enforced == 0
            and enforce.get("energy", {}).get("complete") is False
        ),
    }
    result["analysis_hash"] = "sha256:" + hashlib.sha256(
        canonical_bytes(result)
    ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        result = analyze(args.profile, args.replay)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(result))
    except (AnalysisError, ReplayError) as exc:
        parser.exit(2, f"analysis failed: {exc}\n")
    print(json.dumps({
        "output": str(args.output),
        "analysis_hash": result["analysis_hash"],
        "pass": result["pass"],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
