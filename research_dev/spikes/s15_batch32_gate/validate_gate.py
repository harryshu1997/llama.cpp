#!/usr/bin/env python3
"""Independent replay validator for the S15 independent-phone B32 gate."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PERSIST = ROOT / "research_dev/spikes/s14_mixed_streaming_scheduler/persistence_v2"
ART = PERSIST / "artifacts"
RESULTS = HERE / "results"
REPORT = RESULTS / "gate_report.json"
N_PROCESSES = 7
BATCH = 32
N_GEN = 8
MAX_COV = 0.05
LAYERS = {"op15": 8, "op12": 6}


class ValidationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError(f"unavailable artifact: {path}") from exc
    return digest.hexdigest()


def strict_json(raw: bytes) -> dict:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValidationError(f"invalid JSON constant {value}")

    try:
        value = json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(str(exc)) from exc
    if type(value) is not dict:
        raise ValidationError("JSON record is not an object")
    return value


def records(path: Path, prefix: str) -> list[dict]:
    marker = (prefix + " ").encode("ascii")
    return [strict_json(line[len(marker):]) for line in path.read_bytes().splitlines()
            if line.startswith(marker)]


def validate_manifest() -> None:
    seen = set()
    for line in (ART / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in seen or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValidationError("invalid frozen artifact manifest")
        seen.add(relative)
        if sha256(ART / relative) != digest:
            raise ValidationError(f"frozen artifact mismatch: {relative}")
    if not seen:
        raise ValidationError("empty frozen artifact manifest")


def thermal_ok(value: object, limit: int) -> bool:
    if type(value) is not dict or value.get("valid") is not True:
        return False
    sensors = value.get("sensors_millic")
    maximum = value.get("max_millic")
    return type(sensors) is dict and bool(sensors) and type(maximum) is int \
        and maximum == max(sensors.values()) and maximum <= limit \
        and all(type(item) is int and 10_000 <= item <= 120_000 for item in sensors.values())


def validate_placement(name: str, row: dict, phone_log: Path) -> None:
    certs = records(phone_log, "PLACEMENTCERT")
    if len(certs) != 1 or row.get("placement_cert") != certs[0]:
        raise ValidationError(f"{name}: raw placement binding failed")
    cert = certs[0]
    if cert.get("status") != "SCHEDULED_PLACEMENT_OK" \
            or cert.get("missing_buffer_compute_nodes") != 0 \
            or row.get("placement_status") != cert.get("status") \
            or row.get("missing_buffer_compute_nodes") != 0:
        raise ValidationError(f"{name}: placement status failed")
    mapping = cert.get("compute_by_op_and_buffer")
    if type(mapping) is not dict or not mapping:
        raise ValidationError(f"{name}: placement map is empty")
    htp = 0
    cpu = 0
    for op, buffers in mapping.items():
        if type(buffers) is not dict or not buffers:
            raise ValidationError(f"{name}: placement buffer map is empty")
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0:
                raise ValidationError(f"{name}: placement count is invalid")
            if backend == "HTP0":
                htp += count
            elif backend == "CPU" and op == "GET_ROWS":
                cpu += count
            else:
                raise ValidationError(f"{name}: undeclared {op}@{backend}")
    by_buffer = cert.get("compute_by_buffer_type")
    if type(by_buffer) is not dict or by_buffer.get("HTP0") != htp \
            or by_buffer.get("CPU", 0) != cpu \
            or row.get("compute_htp0_nodes") != htp \
            or row.get("compute_cpu_nodes") != cpu:
        raise ValidationError(f"{name}: placement totals failed")


def validate_record(record: dict, reference: list[int]) -> int:
    name = record.get("device")
    index = record.get("process_index")
    if name not in LAYERS or type(index) is not int or not 0 <= index < N_PROCESSES:
        raise ValidationError("record identity failed")
    if record.get("eligible") is not True or record.get("error") is not None:
        raise ValidationError(f"{name} p{index}: row is not eligible")
    if not thermal_ok(record.get("thermal_start"), 60_000) \
            or not thermal_ok(record.get("thermal_end"), 85_000):
        raise ValidationError(f"{name} p{index}: thermal gate failed")

    expected_files = {
        f"{name}_p{index}/host_k{LAYERS[name]}_b32.stderr",
        f"{name}_p{index}/phone_k{LAYERS[name]}.log",
    }
    artifacts = record.get("log_artifacts")
    if type(artifacts) is not dict or set(artifacts) != expected_files:
        raise ValidationError(f"{name} p{index}: log artifact set failed")
    for relative, digest in artifacts.items():
        path = RESULTS / relative
        if path.resolve().parent.parent != RESULTS.resolve() or sha256(path) != digest:
            raise ValidationError(f"{name} p{index}: log digest failed")

    host_log = RESULTS / f"{name}_p{index}/host_k{LAYERS[name]}_b32.stderr"
    phone_log = RESULTS / f"{name}_p{index}/phone_k{LAYERS[name]}.log"
    route_rows = records(host_log, "ROUTEJSON")
    if len(route_rows) != BATCH \
            or sorted(row.get("stream_index") for row in route_rows) != list(range(BATCH)) \
            or any(row.get("batch_size") != BATCH or row.get("token_ids") != reference
                   for row in route_rows):
        raise ValidationError(f"{name} p{index}: route correctness failed")
    row = record.get("row")
    if type(row) is not dict or row.get("token_ids") != reference \
            or row.get("token_ids_by_request") != [value["token_ids"] for value in route_rows] \
            or row.get("token_match_vs_mono") is not True \
            or row.get("all_tokens_match_vs_mono") is not True \
            or row.get("host_returncode") != 0 \
            or row.get("n_requests_measured") != BATCH \
            or row.get("batch") != BATCH or row.get("n_gen") != N_GEN:
        raise ValidationError(f"{name} p{index}: copied route row failed")
    for raw_name, copied_name in (
        ("request_wall_us", "request_wall_us_p50"),
        ("stage_a_us", "stage_a_us_p50"),
        ("host_us", "host_us_p50"),
    ):
        values = sorted(value.get(raw_name) for value in route_rows)
        if any(type(value) is not int or value <= 0 for value in values) \
                or row.get(copied_name) != values[len(values) // 2]:
            raise ValidationError(f"{name} p{index}: {raw_name} replay failed")
    validate_placement(f"{name} p{index}", row, phone_log)
    return row["request_wall_us_p50"]


def validate_report() -> dict[str, list[int]]:
    report = strict_json(REPORT.read_bytes())
    validate_manifest()
    if report.get("schema") != "s15-independent-b32-gate-v1" \
            or report.get("verdict") != "B32_INDEPENDENT_PHONE_GATE_PASS" \
            or report.get("certified") is not True or report.get("problems") != [] \
            or report.get("scope") != "REAL_DEVICE_B32_CORRECTNESS_PLACEMENT_LATENCY_ONLY_ENERGY_UNKNOWN" \
            or report.get("energy_scope") != "UNKNOWN" \
            or report.get("batch") != BATCH or report.get("n_gen") != N_GEN \
            or report.get("requests_per_process") != BATCH \
            or report.get("processes_requested") != N_PROCESSES:
        raise ValidationError("top-level gate failed")
    if report.get("artifact_manifest_sha256") != sha256(ART / "SHA256SUMS.txt") \
            or report.get("harness_sha256") != sha256(HERE / "run_gate.py"):
        raise ValidationError("harness or artifact-manifest binding failed")
    helper = ROOT / "research_dev/spikes/s14_mixed_streaming_scheduler/energy/stageb_headcert.py"
    if report.get("stageb_helper_sha256") != sha256(helper):
        raise ValidationError("stage helper binding failed")
    model = report.get("model")
    if type(model) is not dict or sha256(Path(model.get("path", ""))) != model.get("sha256"):
        raise ValidationError("model binding failed")

    reference_log = RESULTS / "cuda_b32_reference.log"
    if report.get("reference_log_sha256") != sha256(reference_log):
        raise ValidationError("CUDA reference log binding failed")
    reference_rows = records(reference_log, "ROUTEJSON")
    reference = report.get("reference_tokens")
    if type(reference) is not list or len(reference) != N_GEN \
            or any(type(token) is not int for token in reference) \
            or len(reference_rows) != BATCH \
            or sorted(row.get("stream_index") for row in reference_rows) != list(range(BATCH)) \
            or any(row.get("token_ids") != reference for row in reference_rows):
        raise ValidationError("CUDA reference replay failed")

    raw_records = report.get("records")
    if type(raw_records) is not list or len(raw_records) != 2 * N_PROCESSES:
        raise ValidationError("record count failed")
    expected_order = [(name, index) for index in range(N_PROCESSES) for name in ("op15", "op12")]
    if [(record.get("device"), record.get("process_index")) for record in raw_records] != expected_order:
        raise ValidationError("record order failed")
    walls = {"op15": [], "op12": []}
    for record in raw_records:
        walls[record["device"]].append(validate_record(record, reference))
    profiles = report.get("profiles")
    for name in ("op15", "op12"):
        cov = statistics.pstdev(walls[name]) / statistics.mean(walls[name])
        profile = profiles.get(name) if type(profiles) is dict else None
        if type(profile) is not dict or profile.get("n_processes") != N_PROCESSES \
                or profile.get("request_wall_us_p50_by_process") != walls[name] \
                or type(profile.get("process_cov")) is not float \
                or not math.isclose(profile["process_cov"], cov, abs_tol=1e-15) \
                or cov > MAX_COV:
            raise ValidationError(f"{name}: profile repeatability failed")
    return walls


def main() -> int:
    walls = validate_report()
    print(
        "VALID_B32_GATE "
        f"op15_p50_us={int(statistics.median(walls['op15']))} "
        f"op12_p50_us={int(statistics.median(walls['op12']))}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"INVALID: {exc}")
        raise SystemExit(2)
