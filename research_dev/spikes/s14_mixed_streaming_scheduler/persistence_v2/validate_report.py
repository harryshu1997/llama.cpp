#!/usr/bin/env python3
"""Independent replay of the corrected S15 persistence gate evidence."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
RESULTS = HERE / "results"
N_SESSIONS = 7
N_GEN = 16
MAX_COV = 0.05


class ValidationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_bytes(raw: bytes) -> dict:
    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValidationError(f"duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValidationError(f"invalid JSON constant {value}")

    try:
        result = json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(str(exc)) from exc
    if type(result) is not dict:
        raise ValidationError("top-level JSON must be an object")
    return result


def prefixed(path: Path, prefix: str) -> list[dict]:
    marker = (prefix + " ").encode("ascii")
    result = []
    for line in path.read_bytes().splitlines():
        if line.startswith(marker):
            result.append(load_json_bytes(line[len(marker):]))
    return result


def validate_manifest() -> dict[str, str]:
    expected = {}
    for line in (ART / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in expected:
            raise ValidationError("duplicate manifest path")
        expected[relative] = digest
    for relative, digest in expected.items():
        if sha256(ART / relative) != digest:
            raise ValidationError(f"artifact mismatch {relative}")
    return expected


def thermal_ok(value: object, limit: int) -> bool:
    if type(value) is not dict or value.get("valid") is not True:
        return False
    sensors = value.get("sensors_millic")
    maximum = value.get("max_millic")
    return type(sensors) is dict and bool(sensors) and type(maximum) is int \
        and maximum == max(sensors.values()) and maximum <= limit \
        and all(type(item) is int and 10_000 <= item <= 120_000 for item in sensors.values())


def validate_worker(name: str, report: dict) -> None:
    certs = prefixed(RESULTS / f"worker_{name}.log", "SESSIONCERT")
    if certs != report["worker_certificates"][name] or len(certs) != N_SESSIONS:
        raise ValidationError(f"{name}: certificate binding failed")
    maps = []
    pids = set()
    nonces = set()
    for index, cert in enumerate(certs, 1):
        expected_end = "DETACH" if index < N_SESSIONS else "STOP"
        expected_reset = index < N_SESSIONS
        if cert.get("schema") != "ls-stagenet-session-v2" or cert.get("proto_version") != 2 \
                or cert.get("session_id") != index or cert.get("session_end") != expected_end \
                or cert.get("reset_applied") is not expected_reset \
                or cert.get("steps_session") != 26 or cert.get("steps_total") != index * 26 \
                or cert.get("layer_start") != 0 or cert.get("layer_end") != 6 \
                or cert.get("n_layer") != 48 or cert.get("expected_backend") != "HTP0" \
                or cert.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
                or type(cert.get("missing_buffer_compute_nodes")) is not int \
                or cert["missing_buffer_compute_nodes"] != 0:
            raise ValidationError(f"{name}: session {index} certificate failed")
        mapping = cert.get("compute_by_op_and_buffer")
        if type(mapping) is not dict or not mapping:
            raise ValidationError(f"{name}: session {index} placement missing")
        htp = 0
        for op, buffers in mapping.items():
            if type(buffers) is not dict or not buffers:
                raise ValidationError(f"{name}: invalid placement map")
            for backend, count in buffers.items():
                if type(count) is not int or count <= 0:
                    raise ValidationError(f"{name}: invalid placement count")
                if backend == "HTP0":
                    htp += count
                elif backend != "CPU" or op != "GET_ROWS":
                    raise ValidationError(f"{name}: undeclared {op}@{backend}")
        if htp == 0:
            raise ValidationError(f"{name}: zero HTP compute")
        maps.append(json.dumps(mapping, sort_keys=True))
        pids.add(cert.get("worker_pid"))
        nonces.add(cert.get("worker_boot_nonce"))
    if len(set(maps)) != 1 or len(pids) != 1 or len(nonces) != 1:
        raise ValidationError(f"{name}: worker or placement changed across sessions")


def main() -> int:
    report = load_json_bytes((RESULTS / "gate_report.json").read_bytes())
    manifest = validate_manifest()
    if report.get("schema") != "s15-persistence-gate-v2" \
            or report.get("verdict") != "PERSISTENT_SESSION_REAL_GATE_PASS" \
            or report.get("certified") is not True or report.get("problems") != [] \
            or report.get("energy_scope") != "UNKNOWN":
        raise ValidationError("top-level verdict mismatch")
    if report.get("artifact_manifest_sha256") != sha256(ART / "SHA256SUMS.txt") \
            or report.get("harness_sha256") != sha256(HERE / "run_gate.py"):
        raise ValidationError("manifest or harness binding failed")
    model = report.get("model")
    if type(model) is not dict or sha256(Path(model["path"])) != model.get("sha256"):
        raise ValidationError("model binding failed")
    reference_rows = prefixed(RESULTS / "mono_reference.log", "ROUTEJSON")
    if len(reference_rows) != 1 or reference_rows[0].get("token_ids") != report.get("reference_tokens"):
        raise ValidationError("mono reference binding failed")
    reference = report["reference_tokens"]
    if len(reference) != N_GEN or any(type(token) is not int for token in reference):
        raise ValidationError("reference token vector failed")
    route_walls = []
    sessions = report.get("sessions")
    if type(sessions) is not list or len(sessions) != N_SESSIONS:
        raise ValidationError("session list failed")
    for index, session in enumerate(sessions, 1):
        path = RESULTS / f"host_session_{index}.log"
        rows = prefixed(path, "ROUTEJSON")
        if session.get("rows") != rows or session.get("log_sha256") != sha256(path) \
                or session.get("returncode") != 0 or len(rows) != 2:
            raise ValidationError(f"session {index}: raw host binding failed")
        if sorted(row.get("stream_index") for row in rows) != [0, 1] \
                or any(row.get("status") != "ok" or row.get("token_ids") != reference for row in rows):
            raise ValidationError(f"session {index}: correctness failed")
        walls = [row.get("request_wall_us") for row in rows]
        if any(type(value) is not int or value <= 0 for value in walls):
            raise ValidationError(f"session {index}: invalid route wall")
        route_walls.append(max(walls))
    cov = statistics.pstdev(route_walls) / statistics.mean(route_walls)
    if report.get("route_wall_us") != route_walls \
            or not math.isclose(report.get("route_wall_cov", -1), cov, abs_tol=1e-15) \
            or cov > MAX_COV:
        raise ValidationError("route timing gate failed")
    for name in ("op15", "op12"):
        validate_worker(name, report)
        thermal = report["thermal"][name]
        if not thermal_ok(thermal["start"], 60_000) or not thermal_ok(thermal["end"], 85_000):
            raise ValidationError(f"{name}: thermal gate failed")
        deployed = report["deployments"][name]["files"]
        for filename, digest in deployed.items():
            if manifest.get("android/" + filename) != digest:
                raise ValidationError(f"{name}: deployment binding failed for {filename}")
    print(f"VALID_PERSISTENCE_V2 sessions=7 streams=14 cov={cov:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
