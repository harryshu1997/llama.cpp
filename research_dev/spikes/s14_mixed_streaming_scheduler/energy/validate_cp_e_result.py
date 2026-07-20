#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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
    if type(value) is not dict:
        raise EvidenceError("top level must be an object")
    return value


def percentile(values: list[float], quantile: float) -> float:
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise EvidenceError("invalid latency samples")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = int(rank)
    if lower + 1 == len(ordered):
        return ordered[lower]
    return ordered[lower] + (rank - lower) * (ordered[lower + 1] - ordered[lower])


def replay_energy(record: dict[str, Any]) -> int:
    window = record.get("paid_window_us")
    samples = record.get("raw_power_samples")
    if type(window) is not dict or type(samples) is not list or len(samples) < 2:
        raise EvidenceError("raw power evidence missing")
    start = window.get("start")
    end = window.get("end")
    if type(start) is not int or type(end) is not int or end <= start:
        raise EvidenceError("invalid paid power window")
    energy_nj = 0
    previous = None
    in_window = 0
    max_gap_us = 0
    for sample in samples:
        if type(sample) is not dict:
            raise EvidenceError("invalid power sample")
        t_us = sample.get("t_us")
        power_mw = sample.get("power_mw")
        util = sample.get("util_milli_pct")
        if type(t_us) is not int or type(power_mw) is not int or type(util) is not int or power_mw < 0:
            raise EvidenceError("invalid power sample field")
        if start <= t_us <= end:
            in_window += 1
        if previous is not None:
            if t_us <= previous["t_us"]:
                raise EvidenceError("power timestamps are not increasing")
            lo = max(previous["t_us"], start)
            hi = min(t_us, end)
            if hi > lo:
                energy_nj += previous["power_mw"] * (hi - lo)
                max_gap_us = max(max_gap_us, t_us - previous["t_us"])
        previous = sample
    if samples[0]["t_us"] > start or samples[-1]["t_us"] < end or in_window < 5:
        raise EvidenceError("power samples do not cover the paid window")
    if max_gap_us > 250_000 or max_gap_us != record.get("max_sample_gap_us"):
        raise EvidenceError("power sample gap gate failed")
    if energy_nj != record.get("energy_nj"):
        raise EvidenceError("raw power replay differs from stored energy")
    if not math.isclose(energy_nj / 1_000_000_000, record.get("energy_j", -1), abs_tol=1e-12):
        raise EvidenceError("raw power joules differ from stored energy")
    return energy_nj


def validate_host_cert(cert: object, expected_start: int, expected_end: int) -> None:
    if type(cert) is not dict or cert.get("status") != "SCHEDULED_PLACEMENT_OK":
        raise EvidenceError("host placement status failed")
    if cert.get("missing_buffer_compute_nodes") != 0:
        raise EvidenceError("host placement has missing buffers")
    if cert.get("layer_start") != expected_start or cert.get("layer_end") != expected_end:
        raise EvidenceError("host placement range mismatch")
    by_op = cert.get("compute_by_op_and_buffer")
    if type(by_op) is not dict or not by_op:
        raise EvidenceError("host placement map missing")
    cuda_nodes = 0
    for buffers in by_op.values():
        if type(buffers) is not dict:
            raise EvidenceError("host placement map invalid")
        for buffer, count in buffers.items():
            if type(count) is not int or count <= 0 or "CUDA" not in buffer:
                raise EvidenceError("host compute escaped CUDA")
            cuda_nodes += count
    if cuda_nodes <= 0:
        raise EvidenceError("host CUDA compute missing")


def validate_phone_cert(cert: object, expected_start: int, expected_end: int) -> None:
    if type(cert) is not dict or cert.get("status") != "SCHEDULED_PLACEMENT_OK":
        raise EvidenceError("phone placement status failed")
    if cert.get("missing_buffer_compute_nodes") != 0:
        raise EvidenceError("phone placement has missing buffers")
    if cert.get("layer_start") != expected_start or cert.get("layer_end") != expected_end:
        raise EvidenceError("phone placement range mismatch")
    htp_nodes = 0
    for op, buffers in cert.get("compute_by_op_and_buffer", {}).items():
        for buffer, count in buffers.items():
            if type(count) is not int or count <= 0:
                raise EvidenceError("phone placement count invalid")
            if "HTP" in buffer:
                htp_nodes += count
            elif op != "GET_ROWS":
                raise EvidenceError(f"undeclared phone fallback: {op}@{buffer}")
    if htp_nodes <= 0:
        raise EvidenceError("phone HTP compute missing")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"artifact unavailable: {path}") from exc
    return "sha256:" + digest.hexdigest()


def validate_artifacts(artifacts: object) -> None:
    if type(artifacts) is not dict:
        raise EvidenceError("artifact bundle missing")
    host = artifacts.get("host")
    required = {
        "cp_e_live_priority.py", "bge_server_prof.py", "cp_b_phone_bge.py", "stage_a_gpu_board.py",
        "pipe3_device.py", "stageb_headcert.py", "llama_embedding", "llama_layersplit",
        "android_llama_layersplit", "bge_server_result", "gemma_batch_profile",
    }
    if type(host) is not dict or set(host) != required:
        raise EvidenceError("host artifact set mismatch")
    for name, record in host.items():
        if type(record) is not dict or set(record) != {"path", "sha256"}:
            raise EvidenceError(f"invalid host artifact: {name}")
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise EvidenceError(f"host artifact digest mismatch: {name}")
    android_digest = host["android_llama_layersplit"]["sha256"]
    phone_binary = artifacts.get("phone_binary")
    android_runtime = artifacts.get("android_runtime")
    phone_runtime = artifacts.get("phone_runtime")
    phone_skeleton = artifacts.get("phone_skeleton")
    phone_shards = artifacts.get("phone_shards")
    if type(phone_binary) is not dict or set(phone_binary) != {"op15", "op12"}:
        raise EvidenceError("phone binary records missing")
    if type(phone_shards) is not dict or set(phone_shards) != {"op15", "op12"}:
        raise EvidenceError("phone shard records missing")
    expected_runtime = {
        "libggml-base.so", "libggml-cpu.so", "libggml-hexagon.so", "libggml-opencl.so",
        "libggml.so", "libllama-common.so", "libllama.so",
    }
    if type(android_runtime) is not dict or set(android_runtime) != expected_runtime:
        raise EvidenceError("Android runtime artifact set mismatch")
    for name, record in android_runtime.items():
        if type(record) is not dict or set(record) != {"path", "sha256"} \
                or sha256_file(Path(record["path"])) != record["sha256"]:
            raise EvidenceError(f"Android runtime artifact mismatch: {name}")
    if type(phone_runtime) is not dict or set(phone_runtime) != {"op15", "op12"}:
        raise EvidenceError("phone runtime records missing")
    for device, records in phone_runtime.items():
        if type(records) is not dict or set(records) != expected_runtime:
            raise EvidenceError(f"phone runtime set mismatch: {device}")
        for name, record in records.items():
            if type(record) is not dict or set(record) != {"serial", "remote_path", "sha256"} \
                    or record["sha256"] != android_runtime[name]["sha256"]:
                raise EvidenceError(f"phone runtime digest mismatch: {device}:{name}")
    if type(phone_skeleton) is not dict or set(phone_skeleton) != {"op15", "op12"}:
        raise EvidenceError("phone skeleton records missing")
    for record in phone_skeleton.values():
        digest = record.get("sha256") if type(record) is dict else None
        if type(record) is not dict or set(record) != {"serial", "remote_path", "sha256"} \
                or type(digest) is not str or len(digest) != 71 or not digest.startswith("sha256:"):
            raise EvidenceError("invalid phone skeleton record")
    for records, expected_digest in ((phone_binary, android_digest), (phone_shards, None)):
        for record in records.values():
            if type(record) is not dict or set(record) != {"serial", "remote_path", "sha256"}:
                raise EvidenceError("invalid phone artifact record")
            digest = record["sha256"]
            if type(digest) is not str or len(digest) != 71 or not digest.startswith("sha256:"):
                raise EvidenceError("invalid phone artifact digest")
            if expected_digest is not None and digest != expected_digest:
                raise EvidenceError("deployed phone binary differs from Android build")


def validate(result: dict[str, Any]) -> dict[str, float]:
    if result.get("schema") != "s14-cp-e-live-priority-v1":
        raise EvidenceError("unsupported schema")
    if result.get("second_gpu_idle_at_endpoints") is not True \
            or result.get("experiment_cuda_visible_devices") != result.get("selected_gpu_uuid") \
            or result.get("route") != {
        "op15": [0, 8], "op12": [8, 12], "server": [12, 48],
    }:
        raise EvidenceError("device or route binding failed")
    gates = result.get("gates")
    if type(gates) is not dict:
        raise EvidenceError("gate contract missing")
    high_limit = gates.get("high_priority_p95_ratio_max")
    low_limit = gates.get("low_priority_p95_ratio_max")
    overlap_limit = gates.get("concurrent_overlap_s_min")
    overlap_fraction_limit = gates.get("concurrent_overlap_fraction_of_shorter_min")
    if not all(type(value) is float and math.isfinite(value) and value > 0
               for value in (high_limit, low_limit, overlap_limit, overlap_fraction_limit)):
        raise EvidenceError("invalid gate contract")
    if overlap_fraction_limit > 1.0:
        raise EvidenceError("invalid overlap-fraction gate")
    phone_start_limit = gates.get("phone_start_millic_max")
    phone_end_limit = gates.get("phone_end_millic_max")
    if type(phone_start_limit) is not int or type(phone_end_limit) is not int \
            or not 10_000 <= phone_start_limit <= phone_end_limit <= 120_000:
        raise EvidenceError("invalid phone thermal gate")
    validate_artifacts(result.get("artifacts"))
    batches = result.get("batches")
    if type(batches) is not dict or set(batches) != {"high_priority_bge", "low_priority_gemma"}:
        raise EvidenceError("service batch contract missing")
    bge_batch = batches.get("high_priority_bge")
    gemma_batch = batches.get("low_priority_gemma")
    requests = result.get("requests_per_cohort")
    reference = result.get("reference_tokens")
    workload = result.get("workload")
    if type(workload) is not dict or type(workload.get("bge")) is not dict \
            or type(workload.get("gemma")) is not dict:
        raise EvidenceError("workload contract missing")
    if type(bge_batch) is not int or bge_batch <= 0 or type(gemma_batch) is not int \
            or gemma_batch <= 0 or type(requests) is not int or requests <= 0:
        raise EvidenceError("invalid cohort shape")
    if type(reference) is not list or not reference or not all(type(token) is int for token in reference):
        raise EvidenceError("invalid correctness reference")
    if workload["gemma"].get("n_gen") != len(reference) \
            or workload["gemma"].get("warmup_groups") != 1 \
            or type(workload["gemma"].get("prompt")) is not str:
        raise EvidenceError("Gemma workload differs from the correctness reference")
    rows = result.get("rows")
    if type(rows) is not list or len(rows) != 6:
        raise EvidenceError("expected six measured cohorts")
    groups: dict[str, list[dict[str, Any]]] = {"P0": [], "P2": []}
    bge_work = set()
    gemma_work = set()
    for row in rows:
        label = row.get("label")
        if label not in groups:
            raise EvidenceError("unexpected cohort label")
        groups[label].append(row)
        if row.get("second_gpu_idle_before_go") is not True:
            raise EvidenceError("second GPU was active before cohort GO")
        energy = row.get("selected_gpu_cohort")
        if type(energy) is not dict:
            raise EvidenceError("cohort energy missing")
        replay_energy(energy)
        overlap_s = row.get("concurrent_overlap_s")
        overlap_fraction = row.get("concurrent_overlap_fraction_of_shorter")
        if not isinstance(overlap_s, (int, float)) or isinstance(overlap_s, bool) \
                or not math.isfinite(overlap_s) or overlap_s < 0 \
                or not isinstance(overlap_fraction, (int, float)) or isinstance(overlap_fraction, bool) \
                or not math.isfinite(overlap_fraction) or not 0 <= overlap_fraction <= 1.000001:
            raise EvidenceError("service overlap evidence invalid")
        go_us = int(round(float(row.get("go_time_s")) * 1_000_000))
        if energy["paid_window_us"]["start"] != go_us:
            raise EvidenceError("cohort energy does not start at GO")
        bge = row.get("bge")
        gemma = row.get("gemma")
        if type(bge) is not dict or type(gemma) is not dict:
            raise EvidenceError("service record missing")
        samples = bge.get("latency_samples_us")
        if bge.get("batch") != bge_batch:
            raise EvidenceError("BGE batch differs from the scheduler contract")
        if bge.get("reps") != workload["bge"].get("reps"):
            raise EvidenceError("BGE repetitions differ from the workload contract")
        if bge.get("seq_len_exact") != workload["bge"].get("seq_len_exact") \
                or bge.get("encodes") != bge_batch * bge.get("reps", 0):
            raise EvidenceError("BGE shape or encode count mismatch")
        if type(samples) is not list or len(samples) != bge.get("reps"):
            raise EvidenceError("BGE latency sample count mismatch")
        for key, quantile in (("lat_us_p50", 0.50), ("lat_us_p95", 0.95), ("lat_us_p99", 0.99)):
            if not math.isclose(bge.get(key, -1), percentile(samples, quantile), abs_tol=1e-9):
                raise EvidenceError(f"BGE {key} mismatch")
        placement = bge.get("placement")
        if type(placement) is not dict or placement.get("status") != "SCHEDULED_PLACEMENT_OK":
            raise EvidenceError("BGE placement failed")
        if placement.get("missing_buffer_compute_nodes") != 0:
            raise EvidenceError("BGE placement has missing buffers")
        if not placement.get("by_buffer") or any("CUDA" not in name for name in placement["by_buffer"]):
            raise EvidenceError("BGE compute escaped CUDA")
        if gemma.get("batch") != gemma_batch or gemma.get("requests") != requests \
                or gemma.get("all_tokens_match") is not True:
            raise EvidenceError("Gemma work or correctness mismatch")
        if gemma.get("generated_tokens") != requests * len(reference):
            raise EvidenceError("Gemma generated-token count mismatch")
        validate_host_cert(gemma.get("host_cert"), 0 if label == "P0" else 12, 48)
        if label == "P2":
            if gemma.get("phone_live") is not True:
                raise EvidenceError("P2 lacks live phone execution")
            validate_phone_cert(gemma.get("op15_cert"), 0, 8)
            validate_phone_cert(gemma.get("op12_cert"), 8, 12)
            validate_thermal_pair(row.get("phone_thermal_start"), phone_start_limit)
            validate_thermal_pair(row.get("phone_thermal_end"), phone_end_limit)
        elif gemma.get("phone_live") is not False:
            raise EvidenceError("P0 unexpectedly reports phone execution")
        elif row.get("phone_thermal_start") is not None or row.get("phone_thermal_end") is not None:
            raise EvidenceError("P0 unexpectedly reports phone thermal execution evidence")
        cohort_end_us = int(round(max(float(bge["paid_end_s"]), float(gemma["done_time_s"])) * 1_000_000))
        if energy["paid_window_us"]["end"] != cohort_end_us:
            raise EvidenceError("cohort energy does not end at service completion")
        bge_work.add(bge.get("encodes"))
        gemma_work.add(gemma.get("generated_tokens"))
    for label in groups:
        if sorted(row.get("repeat") for row in groups[label]) != [0, 1, 2]:
            raise EvidenceError(f"{label} repeats incomplete")
    if len(bge_work) != 1 or len(gemma_work) != 1:
        raise EvidenceError("cohorts do not contain matched work")
    p0_energy = statistics.median(row["selected_gpu_cohort"]["energy_j"] for row in groups["P0"])
    p2_energy = statistics.median(row["selected_gpu_cohort"]["energy_j"] for row in groups["P2"])
    p0_p95 = statistics.median(row["bge"]["lat_us_p95"] for row in groups["P0"])
    p2_p95 = statistics.median(row["bge"]["lat_us_p95"] for row in groups["P2"])
    p0_low_p95 = statistics.median(row["gemma"]["lat_us_p95"] for row in groups["P0"])
    p2_low_p95 = statistics.median(row["gemma"]["lat_us_p95"] for row in groups["P2"])
    expected = {
        "p0_selected_gpu_energy_j_median": p0_energy,
        "p2_selected_gpu_energy_j_median": p2_energy,
        "selected_gpu_energy_saving_frac": 1.0 - p2_energy / p0_energy,
        "p0_bge_p95_us_median": p0_p95,
        "p2_bge_p95_us_median": p2_p95,
        "bge_p95_ratio_p2_over_p0": p2_p95 / p0_p95,
        "p0_gemma_p95_us_median": p0_low_p95,
        "p2_gemma_p95_us_median": p2_low_p95,
        "gemma_p95_ratio_p2_over_p0": p2_low_p95 / p0_low_p95,
        "minimum_overlap_s": min(row["concurrent_overlap_s"] for row in rows),
        "minimum_overlap_fraction_of_shorter": min(
            row["concurrent_overlap_fraction_of_shorter"] for row in rows
        ),
    }
    summary = result.get("summary")
    if type(summary) is not dict:
        raise EvidenceError("summary missing")
    for key, value in expected.items():
        if not math.isclose(summary.get(key, math.nan), value, abs_tol=1e-12):
            raise EvidenceError(f"summary mismatch: {key}")
    expected_high = expected["bge_p95_ratio_p2_over_p0"] <= high_limit
    expected_low = expected["gemma_p95_ratio_p2_over_p0"] <= low_limit
    expected_overlap = all(
        row["concurrent_overlap_s"] >= overlap_limit
        and row["concurrent_overlap_fraction_of_shorter"] >= overlap_fraction_limit
        for row in rows
    )
    if summary.get("high_priority_gate") is not expected_high \
            or summary.get("low_priority_gate") is not expected_low \
            or summary.get("overlap_gate") is not expected_overlap:
        raise EvidenceError("stored gate outcome mismatch")
    expected_relief = expected["selected_gpu_energy_saving_frac"] > 0.0
    if summary.get("server_board_relief_gate") is not expected_relief:
        raise EvidenceError("stored energy gate outcome mismatch")
    passed = expected_high and expected_low and expected_overlap and expected_relief
    expected_status = "LIVE_MIXED_PRIORITY_PASS" if passed else "LIVE_MIXED_PRIORITY_FAIL_GATE"
    if result.get("status") != expected_status:
        raise EvidenceError("status does not match measured gates")
    return expected


def validate_thermal_pair(records: object, limit: int) -> None:
    if type(records) is not dict or set(records) != {"op15", "op12"}:
        raise EvidenceError("phone thermal pair missing")
    for name, record in records.items():
        if type(record) is not dict or record.get("valid") is not True:
            raise EvidenceError(f"invalid phone thermal record: {name}")
        sensors = record.get("sensors_millic")
        maximum = record.get("max_millic")
        if type(sensors) is not dict or not sensors or type(maximum) is not int \
                or maximum != max(sensors.values()) or maximum > limit:
            raise EvidenceError(f"phone thermal limit failed: {name}")
        if not all(type(value) is int and 10_000 <= value <= 120_000 for value in sensors.values()):
            raise EvidenceError(f"phone thermal sensor invalid: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    try:
        metrics = validate(load_json(args.result))
    except (EvidenceError, TypeError, ValueError) as exc:
        print(f"INVALID: {exc}")
        return 2
    print(
        "VALID_LIVE selected_gpu_saving="
        f"{metrics['selected_gpu_energy_saving_frac']:.6f} "
        f"bge_p95_ratio={metrics['bge_p95_ratio_p2_over_p0']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
