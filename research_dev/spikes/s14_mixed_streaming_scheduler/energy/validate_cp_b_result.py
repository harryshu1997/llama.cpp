#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


class EvidenceError(ValueError):
    pass


EXPECTED_DEVICES = {"op15": "3C15AU002CL00000", "op12": "5ae7a43d"}
EXPECTED_TARGETS = {32, 128, 512}
EXPECTED_BATCHES = {1, 2, 4}
EXPECTED_CPU_OPS = {"GET_ROWS"}


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


def server_exact(server: dict) -> dict[int, int]:
    rows = server.get("correctness_cuda_vs_cpu")
    if not isinstance(rows, dict) or not rows:
        raise EvidenceError("server correctness rows missing")
    return {int(target): row["seq_len_exact"] for target, row in rows.items()}


def validate(phone: dict, server: dict, persisted_server_digest: str) -> list[str]:
    issues = []
    expected = server_exact(server)
    if phone.get("schema") != "s14-cp-b-phone-bge-atlas-v2":
        issues.append("E_SCHEMA")
    if set(expected) != EXPECTED_TARGETS:
        issues.append("E_SERVER_TARGETS")
    if type(persisted_server_digest) is not str or len(persisted_server_digest) != 64:
        raise EvidenceError("persisted server digest is invalid")
    if phone.get("server_profile_sha256") != persisted_server_digest:
        issues.append("E_SERVER_DIGEST")
    bench = phone.get("bench", {})
    procs = bench.get("procs")
    reps = bench.get("reps_per_proc")
    if isinstance(procs, bool) or not isinstance(procs, int) or procs < 7:
        issues.append("E_PROCS")
        procs = 0
    if isinstance(reps, bool) or not isinstance(reps, int) or reps <= 0:
        issues.append("E_REPS")
        reps = 0
    if phone.get("n_failures") != 0 or phone.get("failures"):
        issues.append("E_REPORTED_FAILURE")
    devices = phone.get("devices", {})
    if type(devices) is not dict or set(devices) != set(EXPECTED_DEVICES):
        issues.append("E_DEVICE_SET")
        devices = devices if type(devices) is dict else {}
    binary_digests = set()
    for name, device in devices.items():
        if type(device) is not dict:
            issues.append(f"E_DEVICE_RECORD:{name}")
            continue
        if device.get("serial") != EXPECTED_DEVICES.get(name):
            issues.append(f"E_DEVICE_SERIAL:{name}")
        for field in ("binary_sha256", "skel_sha256", "model_sha256"):
            digest = device.get(field)
            if type(digest) is not str or len(digest) != 64 \
                    or any(char not in "0123456789abcdef" for char in digest):
                issues.append(f"E_DIGEST:{name}:{field}")
        binary_digests.add(device.get("binary_sha256"))
        start = device.get("thermal_npu_start")
        end = device.get("thermal_npu_end")
        if not thermal_valid(start, phone.get("thermal_max_millic")) \
                or not thermal_valid(end, phone.get("thermal_max_millic")):
            issues.append(f"E_THERMAL:{name}")
    if len(binary_digests) != 1:
        issues.append("E_BINARY_IDENTITY")
    seen = set()
    for row in phone.get("shapes", []):
        if type(row) is not dict:
            issues.append("E_ROW_TYPE")
            continue
        key = (row.get("device"), row.get("seq_len_target"), row.get("batch"))
        if not all(type(value) in {str, int} for value in key):
            issues.append("E_ROW_KEY")
            continue
        if key in seen:
            issues.append(f"E_DUPLICATE_ROW:{key}")
        seen.add(key)
        target = row.get("seq_len_target")
        if target not in expected or row.get("seq_len_exact") != expected.get(target):
            issues.append(f"E_SHAPE_MATCH:{key}")
        if row.get("ok_procs") != procs or row.get("n_samples") != procs * reps:
            issues.append(f"E_SAMPLE_COUNT:{key}")
        cov = row.get("cov")
        if not isinstance(cov, (int, float)) or isinstance(cov, bool) \
                or not math.isfinite(cov) or cov < 0 or cov > 0.05:
            issues.append(f"E_COV:{key}")
        p50 = row.get("lat_us_p50")
        p95 = row.get("lat_us_p95")
        p99 = row.get("lat_us_p99")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) and value > 0 for value in (p50, p95, p99)) \
                or not p50 <= p95 <= p99:
            issues.append(f"E_LATENCY:{key}")
        if row.get("cert_status") != "SCHEDULED_PLACEMENT_OK":
            issues.append(f"E_PLACEMENT:{key}")
        by_buffer = row.get("cert_compute_by_buffer")
        if type(by_buffer) is not dict or sum(
            count for name, count in by_buffer.items()
            if "HTP" in name and type(count) is int and count > 0
        ) <= 0:
            issues.append(f"E_HTP_NODES:{key}")
        non_htp = row.get("cert_non_htp_ops")
        if type(non_htp) is not list or any(
            type(entry) is not str or entry.split("@", 1)[0] not in EXPECTED_CPU_OPS
            for entry in non_htp
        ):
            issues.append(f"E_FALLBACK:{key}")
        cosine = row.get("cosine_vs_cpu")
        if not isinstance(cosine, (int, float)) or isinstance(cosine, bool) \
                or not math.isfinite(cosine) or cosine < 0.99 or cosine > 1.000001:
            issues.append(f"E_CORRECTNESS:{key}")
    expected_rows = {
        (device, target, batch)
        for device in EXPECTED_DEVICES for target in EXPECTED_TARGETS for batch in EXPECTED_BATCHES
    }
    if seen != expected_rows:
        issues.append("E_MATRIX")
    return issues


def thermal_valid(snapshot: object, limit: object) -> bool:
    if not isinstance(snapshot, dict) or snapshot.get("valid") is not True:
        return False
    sensors = snapshot.get("sensors_millic")
    maximum = snapshot.get("max_millic")
    if not isinstance(sensors, dict) or not sensors or type(maximum) is not int:
        return False
    if type(limit) is not int or limit <= 0 or maximum > limit:
        return False
    return maximum == max(sensors.values()) and all(
        type(value) is int and 10_000 <= value <= 120_000 for value in sensors.values()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phone", type=Path)
    parser.add_argument("server", type=Path)
    args = parser.parse_args()
    try:
        phone = load_json(args.phone)
        server = load_json(args.server)
        server_digest = hashlib.sha256(args.server.read_bytes()).hexdigest()
        issues = validate(phone, server, server_digest)
    except (EvidenceError, TypeError, ValueError) as exc:
        print(f"INVALID: {exc}")
        return 2
    if issues:
        print("INELIGIBLE")
        for issue in issues:
            print(issue)
        return 2
    print("ELIGIBLE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
