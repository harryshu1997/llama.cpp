#!/usr/bin/env python3
"""Reopen S17 CP1 raw tensors and reproduce the fail-closed gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from array import array
from pathlib import Path
from typing import Any


N_EMBD = 3840
ROWS = 64
EXPECTED_BYTES = ROWS * N_EMBD * 4
ALLOWED_CPU_OPS = {"GET_ROWS"}


class ValidationError(RuntimeError):
    pass


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"), object_pairs_hook=strict_object)
    if type(value) is not dict:
        raise ValidationError("report root is not an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(report_path: Path, entry: dict[str, Any]) -> Path:
    raw = entry.get("path")
    if type(raw) is not str or not raw:
        raise ValidationError("artifact path is missing")
    path = Path(raw)
    if path.is_file():
        return path
    candidate = report_path.parent / path.name
    if candidate.is_file():
        return candidate
    source_candidate = report_path.parent / "executed-source" / path.name
    if source_candidate.is_file():
        return source_candidate
    raise ValidationError(f"artifact is missing: {raw}")


def validate_artifact(report_path: Path, entry: dict[str, Any]) -> Path:
    if type(entry) is not dict or set(entry) != {"path", "bytes", "sha256"}:
        raise ValidationError("invalid artifact record")
    path = resolve_artifact(report_path, entry)
    if type(entry["bytes"]) is not int or entry["bytes"] != path.stat().st_size:
        raise ValidationError(f"artifact size mismatch: {path}")
    if type(entry["sha256"]) is not str or entry["sha256"] != sha256_file(path):
        raise ValidationError(f"artifact digest mismatch: {path}")
    return path


def read_f32(path: Path, rows: int = ROWS) -> array:
    expected_bytes = rows * N_EMBD * 4
    if path.stat().st_size != expected_bytes:
        raise ValidationError(f"wrong tensor size: {path}")
    values = array("f")
    values.frombytes(path.read_bytes())
    if len(values) != rows * N_EMBD or not all(math.isfinite(value) for value in values):
        raise ValidationError(f"invalid tensor values: {path}")
    return values


def rel_l2(values: array, reference: array) -> float:
    diff2 = 0.0
    ref2 = 0.0
    for value, ref in zip(values, reference):
        delta = float(value) - float(ref)
        diff2 += delta * delta
        ref2 += float(ref) * float(ref)
    return math.sqrt(diff2 / ref2) if ref2 > 0 else math.inf


def argmax_mismatches(values: array, reference: array) -> int:
    mismatches = 0
    for row in range(ROWS):
        begin = row * N_EMBD
        end = begin + N_EMBD
        lhs = max(range(begin, end), key=values.__getitem__)
        rhs = max(range(begin, end), key=reference.__getitem__)
        mismatches += lhs != rhs
    return mismatches


def placement_ok(cert: Any, backend: str, start: int, end: int) -> bool:
    if type(cert) is not dict or cert.get("status") != "SCHEDULED_PLACEMENT_OK" or \
            cert.get("layer_start") != start or cert.get("layer_end") != end or \
            cert.get("missing_buffer_compute_nodes") != 0:
        return False
    mapping = cert.get("compute_by_op_and_buffer")
    if type(mapping) is not dict:
        return False
    backend_nodes = 0
    for op, buffers in mapping.items():
        if type(op) is not str or type(buffers) is not dict:
            return False
        for buffer, count in buffers.items():
            if type(buffer) is not str or type(count) is not int or count <= 0:
                return False
            if backend in buffer:
                backend_nodes += count
            elif op not in ALLOWED_CPU_OPS:
                return False
    return backend_nodes > 0


def close_float(actual: Any, expected: float) -> bool:
    return type(actual) is float and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def validate(report_path: Path) -> dict[str, Any]:
    report = load_json(report_path)
    if report.get("schema") != "s17-real-boundary-middle-screen-v1":
        raise ValidationError("wrong report schema")
    if report.get("status") not in {"CP1_REAL_BOUNDARY_PASS", "CP1_REAL_BOUNDARY_FAIL"}:
        raise ValidationError("invalid report status")

    artifacts = report.get("artifacts")
    identity = report.get("identity")
    if type(artifacts) is not dict or type(identity) is not dict:
        raise ValidationError("artifact binding is missing")
    for entry in artifacts.values():
        validate_artifact(report_path, entry)
    for name in ("harness", "middle_harness", "host_binary", "middle_shard"):
        validate_artifact(report_path, identity.get(name))

    raw = {}
    for name in ("htp.b64.f32", "htp.split32.f32", "cuda.b64.f32", "cuda.split32.f32"):
        raw[name] = read_f32(report_path.parent / name)
    htp_l2 = rel_l2(raw["htp.b64.f32"], raw["htp.split32.f32"])
    htp_argmax = argmax_mismatches(raw["htp.b64.f32"], raw["htp.split32.f32"])
    cross_l2 = rel_l2(raw["htp.b64.f32"], raw["cuda.b64.f32"])
    cross_argmax = argmax_mismatches(raw["htp.b64.f32"], raw["cuda.b64.f32"])

    heads = report.get("heads")
    middles = report.get("middles")
    if type(heads) is not dict or set(heads) != {"op12", "op15"} or \
            type(middles) is not dict or set(middles) != {"cuda", "htp"}:
        raise ValidationError("worker set is incomplete")
    for name in ("op12", "op15"):
        read_f32(report_path.parent / f"{name}-head.f32", 32)

    htp = middles["htp"]
    cuda = middles["cuda"]
    if not close_float(htp.get("b64_vs_split_rel_l2"), htp_l2) or \
            htp.get("b64_vs_split_argmax_mismatches") != htp_argmax:
        raise ValidationError("stored HTP comparison does not match raw tensors")
    cross = report.get("cross_backend")
    if type(cross) is not dict or not close_float(cross.get("b64_rel_l2"), cross_l2) or \
            cross.get("b64_argmax_mismatches") != cross_argmax:
        raise ValidationError("stored cross-backend comparison does not match raw tensors")

    checks = {
        "head_workers_complete": all(
            head.get("returncode") == 0 and head.get("error") is None for head in heads.values()
        ),
        "head_placement_valid": all(
            placement_ok(head.get("placement"), "HTP0", 0, 6) for head in heads.values()
        ),
        "head_outputs_finite": all(head.get("finite") is True for head in heads.values()),
        "middle_workers_complete": all(
            middle.get("returncode") == 0 and middle.get("error") is None
            for middle in middles.values()
        ),
        "middle_placement_valid": placement_ok(htp.get("placement"), "HTP0", 6, 12) and
            placement_ok(cuda.get("placement"), "CUDA0", 6, 12),
        "middle_outputs_finite": all(middle.get("finite") is True for middle in middles.values()),
        "middle_repeat_stable": all(
            type(middle.get("repeat_rel_l2_max")) is float and
            middle["repeat_rel_l2_max"] <= 5e-3 for middle in middles.values()
        ),
        "htp_b64_vs_2xb32": htp_l2 <= 5e-3 and htp_argmax == 0,
        "htp_vs_cuda_b64": cross_l2 <= 5e-3 and cross_argmax == 0,
        "htp_coalescing_faster": (
            type(htp.get("b64_us")) is list and type(htp.get("split32_us")) is list and
            len(htp["b64_us"]) >= 3 and len(htp["split32_us"]) >= 3 and
            statistics.median(htp["split32_us"]) >= statistics.median(htp["b64_us"])
        ),
        "head_shards_identical": all(
            type(head.get("shard_sha256")) is str and len(head["shard_sha256"]) == 64
            for head in heads.values()
        ) and len({head["shard_sha256"] for head in heads.values()}) == 1,
    }
    if report.get("checks") != checks:
        raise ValidationError("stored checks do not match recomputation")
    expected_status = "CP1_REAL_BOUNDARY_PASS" if all(checks.values()) else "CP1_REAL_BOUNDARY_FAIL"
    if report["status"] != expected_status:
        raise ValidationError("stored verdict does not match recomputation")
    return {
        "validated_status": expected_status,
        "htp_b64_vs_2xb32_rel_l2": htp_l2,
        "htp_vs_cuda_b64_rel_l2": cross_l2,
        "htp_vs_cuda_b64_argmax_mismatches": cross_argmax,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.report)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"INVALID: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
