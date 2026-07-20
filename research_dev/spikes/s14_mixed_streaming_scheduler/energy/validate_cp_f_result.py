#!/usr/bin/env python3
"""Independent replay for the S14 CP-F live OP15 result."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import validate_cp_e_result as VE

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
sys.path.insert(0, str(SPIKE))

from live_profile_adapter import load_op15_head_route  # noqa: E402


class ValidationError(ValueError):
    pass


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        result = json.loads(path.read_bytes(), object_pairs_hook=no_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(str(exc)) from exc
    if type(result) is not dict:
        raise ValidationError("top level must be an object")
    return result


def check_artifacts(result: dict) -> str:
    artifacts = result.get("artifacts")
    if type(artifacts) is not dict:
        raise ValidationError("artifact bundle missing")
    host = artifacts.get("host")
    profiles = artifacts.get("op15_profile_processes")
    if type(host) is not dict or type(profiles) is not dict or len(profiles) != 7:
        raise ValidationError("host or seven-process profile artifacts missing")
    for records in (host, profiles):
        for name, record in records.items():
            if type(record) is not dict or set(record) != {"path", "sha256"}:
                raise ValidationError(f"invalid artifact record {name}")
            if digest(Path(record["path"])) != record["sha256"]:
                raise ValidationError(f"artifact digest mismatch {name}")
    android = host.get("llama_layersplit_android", {}).get("sha256")
    if artifacts.get("phone_binary") != android:
        raise ValidationError("deployed OP15 binary differs from Android build")
    shard = artifacts.get("phone_shard")
    if type(shard) is not str or not shard.startswith("sha256:") or len(shard) != 71:
        raise ValidationError("phone shard digest invalid")
    paths = [Path(record["path"]) for record in profiles.values()]
    return load_op15_head_route(paths, 1).profile_id


def check_thermal(value: object, limit: int) -> None:
    if type(value) is not dict or set(value) != {"op15"}:
        raise ValidationError("OP15 thermal record missing")
    record = value["op15"]
    if type(record) is not dict or record.get("valid") is not True:
        raise ValidationError("invalid OP15 thermal record")
    sensors = record.get("sensors_millic")
    maximum = record.get("max_millic")
    if type(sensors) is not dict or not sensors or type(maximum) is not int \
            or maximum != max(sensors.values()) or maximum > limit:
        raise ValidationError("OP15 thermal gate failed")


def validate(result: dict) -> dict[str, float]:
    if result.get("schema") != "s14-cp-f-live-op15-v1":
        raise ValidationError("unsupported schema")
    if result.get("selected_gpu_uuid") != "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f" \
            or result.get("second_gpu_idle_at_endpoints") is not True \
            or result.get("route") != {"op15": [0, 8], "server": [8, 48]}:
        raise ValidationError("device or route binding failed")
    profile = result.get("profile")
    if type(profile) is not dict or profile.get("op15_batch") != 1:
        raise ValidationError("profile binding missing")
    if check_artifacts(result) != profile.get("op15"):
        raise ValidationError("OP15 route profile digest mismatch")
    bge_path = Path(result["artifacts"]["host"]["bge_server_profile"]["path"])
    if digest(bge_path) != profile.get("bge"):
        raise ValidationError("BGE profile digest mismatch")
    workload = result.get("workload")
    reference = result.get("reference_tokens")
    if type(workload) is not dict or workload.get("gemma_batch") != 1 \
            or type(reference) is not list or not reference:
        raise ValidationError("workload or reference missing")
    rows = result.get("rows")
    if type(rows) is not list or len(rows) != 6:
        raise ValidationError("expected six cohorts")
    groups = {"P0": [], "P2": []}
    bge_work = set()
    gemma_work = set()
    for row in rows:
        label = row.get("label")
        if label not in groups or row.get("second_gpu_idle_before_go") is not True:
            raise ValidationError("invalid cohort label or second-GPU gate")
        groups[label].append(row)
        VE.replay_energy(row.get("selected_gpu_cohort"))
        bge = row.get("bge")
        gemma = row.get("gemma")
        if type(bge) is not dict or type(gemma) is not dict:
            raise ValidationError("service evidence missing")
        samples = bge.get("latency_samples_us")
        if type(samples) is not list or len(samples) != workload.get("bge_reps") \
                or bge.get("batch") != workload.get("bge_batch"):
            raise ValidationError("BGE work mismatch")
        for key, quantile in (("lat_us_p50", 0.50), ("lat_us_p95", 0.95), ("lat_us_p99", 0.99)):
            if not math.isclose(bge.get(key, -1), VE.percentile(samples, quantile), abs_tol=1e-9):
                raise ValidationError(f"BGE {key} mismatch")
        placement = bge.get("placement")
        if type(placement) is not dict or placement.get("status") != "SCHEDULED_PLACEMENT_OK" \
                or placement.get("missing_buffer_compute_nodes") != 0:
            raise ValidationError("BGE placement failed")
        if gemma.get("batch") != 1 or gemma.get("requests") != workload.get("gemma_requests") \
                or gemma.get("generated_tokens") != workload.get("gemma_requests") * len(reference) \
                or gemma.get("all_tokens_match") is not True:
            raise ValidationError("Gemma work or correctness mismatch")
        VE.validate_host_cert(gemma.get("host_cert"), 0 if label == "P0" else 8, 48)
        if label == "P2":
            VE.validate_phone_cert(gemma.get("op15_cert"), 0, 8)
            check_thermal(row.get("phone_thermal_start"), 60_000)
            check_thermal(row.get("phone_thermal_end"), 85_000)
        elif gemma.get("phone_live") is not False \
                or row.get("phone_thermal_start") is not None \
                or row.get("phone_thermal_end") is not None:
            raise ValidationError("P0 contains phone execution evidence")
        overlap_start = max(float(row["go_time_s"]), float(bge["paid_start_s"]))
        overlap_end = min(float(bge["paid_end_s"]), float(gemma["done_time_s"]))
        overlap = max(0.0, overlap_end - overlap_start)
        shorter = min(
            float(bge["paid_end_s"]) - float(bge["paid_start_s"]),
            float(gemma["done_time_s"]) - float(row["go_time_s"]),
        )
        fraction = overlap / shorter if shorter > 0 else 0.0
        if not math.isclose(row.get("concurrent_overlap_s", -1), overlap, abs_tol=1e-12) \
                or not math.isclose(
                    row.get("concurrent_overlap_fraction_of_shorter", -1), fraction, abs_tol=1e-12
                ):
            raise ValidationError("overlap replay mismatch")
        bge_work.add(bge.get("encodes"))
        gemma_work.add(gemma.get("generated_tokens"))
    if any(sorted(row["repeat"] for row in group) != [0, 1, 2] for group in groups.values()) \
            or len(bge_work) != 1 or len(gemma_work) != 1:
        raise ValidationError("repeat or matched-work gate failed")
    p0_energy = statistics.median(row["selected_gpu_cohort"]["energy_j"] for row in groups["P0"])
    p2_energy = statistics.median(row["selected_gpu_cohort"]["energy_j"] for row in groups["P2"])
    p0_high = statistics.median(row["bge"]["lat_us_p95"] for row in groups["P0"])
    p2_high = statistics.median(row["bge"]["lat_us_p95"] for row in groups["P2"])
    p0_low = statistics.median(row["gemma"]["lat_us_p95"] for row in groups["P0"])
    p2_low = statistics.median(row["gemma"]["lat_us_p95"] for row in groups["P2"])
    expected = {
        "selected_gpu_energy_saving_frac": 1.0 - p2_energy / p0_energy,
        "bge_p95_ratio_p2_over_p0": p2_high / p0_high,
        "gemma_p95_ratio_p2_over_p0": p2_low / p0_low,
    }
    summary = result.get("summary")
    if type(summary) is not dict:
        raise ValidationError("summary missing")
    for key, value in expected.items():
        if not math.isclose(summary.get(key, math.nan), value, abs_tol=1e-12):
            raise ValidationError(f"summary mismatch {key}")
    passed = expected["bge_p95_ratio_p2_over_p0"] <= 1.05 \
        and expected["gemma_p95_ratio_p2_over_p0"] <= 2.0 \
        and summary.get("overlap_gate") is True \
        and expected["selected_gpu_energy_saving_frac"] > 0
    if result.get("status") != ("LIVE_OP15_PASS" if passed else "LIVE_OP15_FAIL_GATE"):
        raise ValidationError("status differs from replayed gates")
    return expected


def main() -> int:
    try:
        metrics = validate(load(Path(sys.argv[1])))
    except (IndexError, OSError, TypeError, ValueError, VE.EvidenceError) as exc:
        print(f"INVALID: {exc}")
        return 2
    print(
        "VALID_LIVE_OP15 "
        f"gpu_saving={metrics['selected_gpu_energy_saving_frac']:.6f} "
        f"high_ratio={metrics['bge_p95_ratio_p2_over_p0']:.6f} "
        f"low_ratio={metrics['gemma_p95_ratio_p2_over_p0']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
