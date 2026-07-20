#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


class EvidenceError(ValueError):
    pass


def load_json(path: Path) -> dict:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=no_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(str(exc)) from exc
    if not isinstance(value, dict):
        raise EvidenceError("top level must be an object")
    return value


def validate(result: dict) -> float:
    schema = result.get("schema")
    if schema not in {"s14-cp-d-priority-v2", "s14-cp-d-priority-v3"}:
        raise EvidenceError("unsupported schema")
    expected_status = (
        "GPU_BOARD_COMPUTE_WINDOW_P0_P2_DIAGNOSTIC_PASS"
        if schema == "s14-cp-d-priority-v3"
        else "GPU_BOARD_COMPUTE_WINDOW_P0_P2_PASS"
    )
    if result.get("status") != expected_status:
        raise EvidenceError("result status is not PASS")
    if result.get("second_gpu_idle") is not True:
        raise EvidenceError("second GPU was not idle")
    decisions = result.get("policy_decisions", {})
    profiles = result.get("profiles", {})
    if decisions.get("high", {}).get("batch") != profiles.get("bge_batch"):
        raise EvidenceError("BGE policy/profile batch mismatch")
    if decisions.get("low", {}).get("batch") != profiles.get("gemma_batch"):
        raise EvidenceError("Gemma policy/profile batch mismatch")
    rows = result.get("measured", {}).get("plan_rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise EvidenceError("expected six plan rows")
    if schema == "s14-cp-d-priority-v3":
        validate_artifacts(result.get("artifacts"))
    matched = result.get("matched_work_per_plan", {})
    by_label = {}
    for row in rows:
        label = row.get("label")
        if label not in {"P0", "P2"}:
            raise EvidenceError("unexpected plan label")
        by_label.setdefault(label, []).append(row)
        bge = row.get("bge", {})
        gemma = row.get("gemma", {})
        if bge.get("encodes") != matched.get("bge_encodes"):
            raise EvidenceError("BGE work mismatch")
        if gemma.get("tokens") != matched.get("gemma_decode_tokens"):
            raise EvidenceError("Gemma work mismatch")
        if bge.get("n_samples", 0) < 5 or gemma.get("n_samples", 0) < 5:
            raise EvidenceError("too few power samples")
        if bge.get("max_sample_gap_s", 1.0) > 0.25 or gemma.get("max_sample_gap_s", 1.0) > 0.25:
            raise EvidenceError("power sample gap too large")
        placement = bge.get("placement", {})
        if placement.get("status") != "SCHEDULED_PLACEMENT_OK":
            raise EvidenceError("BGE placement failed")
        if placement.get("missing_buffer_compute_nodes") != 0:
            raise EvidenceError("BGE placement has missing buffers")
        if any("CUDA" not in name for name in placement.get("by_buffer", {})):
            raise EvidenceError("BGE placement escaped CUDA")
        expected = bge.get("energy_j", 0.0) + gemma.get("energy_j", 0.0)
        if not math.isclose(row.get("selected_gpu_energy_j", -1.0), expected, rel_tol=0.0, abs_tol=1e-9):
            raise EvidenceError("selected GPU energy does not sum paid legs")
        if gemma.get("k") != (0 if label == "P0" else 8):
            raise EvidenceError("wrong Gemma layer boundary")
        if schema == "s14-cp-d-priority-v3":
            replay_energy(bge)
            replay_energy(gemma)
    for label in ("P0", "P2"):
        if sorted(row.get("repeat") for row in by_label.get(label, [])) != [0, 1, 2]:
            raise EvidenceError(f"{label} repeats incomplete")
    p0 = statistics.median(row["selected_gpu_energy_j"] for row in by_label["P0"])
    p2 = statistics.median(row["selected_gpu_energy_j"] for row in by_label["P2"])
    saving = 1.0 - p2 / p0
    measured = result["measured"]
    if not math.isclose(measured.get("p0_selected_gpu_energy_j_median", -1.0), p0, abs_tol=1e-9):
        raise EvidenceError("P0 median mismatch")
    if not math.isclose(measured.get("p2_selected_gpu_energy_j_median", -1.0), p2, abs_tol=1e-9):
        raise EvidenceError("P2 median mismatch")
    if not math.isclose(measured.get("selected_gpu_energy_saving_frac", -1.0), saving, abs_tol=1e-12):
        raise EvidenceError("saving mismatch")
    return saving


def validate_artifacts(artifacts: object) -> None:
    required = {
        "cp_d_priority.py",
        "bge_server_prof.py",
        "stage_a_gpu_board.py",
        "llama_embedding",
        "llama_layersplit",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise EvidenceError("artifact set mismatch")
    for name, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise EvidenceError(f"invalid artifact record: {name}")
        path = Path(record["path"])
        try:
            digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, TypeError) as exc:
            raise EvidenceError(f"artifact unavailable: {name}") from exc
        if digest != record["sha256"]:
            raise EvidenceError(f"artifact digest mismatch: {name}")


def replay_energy(leg: dict) -> int:
    window = leg.get("paid_window_us")
    samples = leg.get("raw_power_samples")
    if not isinstance(window, dict) or not isinstance(samples, list) or len(samples) < 2:
        raise EvidenceError("raw power evidence missing")
    start = window.get("start")
    end = window.get("end")
    if type(start) is not int or type(end) is not int or end <= start:
        raise EvidenceError("invalid raw power window")
    previous = None
    energy_nj = 0
    for row in samples:
        t_us = row.get("t_us")
        power_mw = row.get("power_mw")
        if type(t_us) is not int or type(power_mw) is not int or power_mw < 0:
            raise EvidenceError("invalid raw power sample")
        if previous is not None:
            if t_us <= previous["t_us"]:
                raise EvidenceError("raw power timestamps are not increasing")
            lo = max(previous["t_us"], start)
            hi = min(t_us, end)
            if hi > lo:
                energy_nj += previous["power_mw"] * (hi - lo)
        previous = row
    if samples[0]["t_us"] > start or samples[-1]["t_us"] < end:
        raise EvidenceError("raw power samples do not bracket the window")
    if energy_nj != leg.get("energy_nj"):
        raise EvidenceError("raw power replay differs from stored energy")
    if not math.isclose(energy_nj / 1_000_000_000, leg.get("energy_j", -1.0), abs_tol=1e-12):
        raise EvidenceError("raw power joules differ from stored energy")
    return energy_nj


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    try:
        result = load_json(args.result)
        saving = validate(result)
    except EvidenceError as exc:
        print(f"INVALID: {exc}")
        return 2
    scope = "RAW_REPLAY" if result.get("schema") == "s14-cp-d-priority-v3" else "SUMMARY_ONLY"
    print(f"VALID_{scope} selected_gpu_paid_window_saving={saving:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
