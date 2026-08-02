#!/usr/bin/env python3
"""Compare S24 tokens and boundary activations against a physical control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from array import array
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "s24-physical-comparison-v1"
RUN_SCHEMA = "s24-fixed-diamond-physical-v1"


class ComparisonError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot load run {path}: {exc}") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != RUN_SCHEMA
        or value.get("status") != "RUN_COMPLETE"
    ):
        raise ComparisonError(f"run is not complete: {path}")
    requests = value.get("runtime", {}).get("requests")
    if not isinstance(requests, list) or not requests:
        raise ComparisonError(f"run has no completed requests: {path}")
    return value


def read_activation(record: dict[str, Any]) -> tuple[float, ...]:
    expected = {
        "worker", "layer_end", "position", "elements", "bytes", "sha256",
        "l2_norm", "minimum", "maximum", "path",
    }
    if not isinstance(record, dict) or set(record) != expected:
        raise ComparisonError("activation record contract mismatch")
    path = Path(record["path"])
    payload = path.read_bytes()
    if len(payload) != record["bytes"] or len(payload) != 4 * record["elements"]:
        raise ComparisonError(f"activation size mismatch: {path}")
    observed = "sha256:" + hashlib.sha256(payload).hexdigest()
    if observed != record["sha256"]:
        raise ComparisonError(f"activation hash mismatch: {path}")
    values = array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != record["elements"] or not all(math.isfinite(value) for value in values):
        raise ComparisonError(f"activation values are invalid: {path}")
    return tuple(values)


def vector_terms(
    reference: Sequence[float], candidate: Sequence[float],
) -> dict[str, float]:
    if len(reference) != len(candidate) or not reference:
        raise ComparisonError("activation vector shapes differ")
    diff2 = 0.0
    ref2 = 0.0
    cand2 = 0.0
    dot = 0.0
    max_abs = 0.0
    for ref, cand in zip(reference, candidate):
        if not math.isfinite(ref) or not math.isfinite(cand):
            raise ComparisonError("activation vector is non-finite")
        delta = float(cand) - float(ref)
        diff2 += delta * delta
        ref2 += float(ref) * float(ref)
        cand2 += float(cand) * float(cand)
        dot += float(ref) * float(cand)
        max_abs = max(max_abs, abs(delta))
    rel_l2 = math.sqrt(diff2 / ref2) if ref2 > 0 else (
        0.0 if diff2 == 0 else math.inf
    )
    cosine = dot / math.sqrt(ref2 * cand2) if ref2 > 0 and cand2 > 0 else (
        1.0 if diff2 == 0 else 0.0
    )
    return {
        "diff2": diff2,
        "reference2": ref2,
        "candidate2": cand2,
        "dot": dot,
        "relative_l2": rel_l2,
        "cosine": cosine,
        "max_abs": max_abs,
    }


def boundary_map(request: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    records = request.get("boundary_activations")
    if not isinstance(records, list) or not records:
        raise ComparisonError("request has no captured boundary activations")
    result = {}
    for record in records:
        key = (int(record["layer_end"]), int(record["position"]))
        if key in result:
            raise ComparisonError("duplicate boundary layer and position")
        result[key] = record
    return result


def compare_runs(
    reference_path: Path,
    candidate_path: Path,
    label: str,
    relative_l2_limit: float,
) -> dict[str, Any]:
    if not label:
        raise ComparisonError("comparison label must be nonempty")
    if not 0.0 < relative_l2_limit < 1.0:
        raise ComparisonError("relative-L2 limit is invalid")
    reference = load_run(reference_path)
    candidate = load_run(candidate_path)
    reference_requests = {
        row["request_id"]: row for row in reference["runtime"]["requests"]
    }
    candidate_requests = {
        row["request_id"]: row for row in candidate["runtime"]["requests"]
    }
    if set(reference_requests) != set(candidate_requests):
        raise ComparisonError("control and candidate request identities differ")

    request_results = []
    total_diff2 = 0.0
    total_ref2 = 0.0
    total_cand2 = 0.0
    total_dot = 0.0
    maximum_abs = 0.0
    all_tokens_equal = True
    for request_id in sorted(reference_requests):
        ref_request = reference_requests[request_id]
        cand_request = candidate_requests[request_id]
        if (
            ref_request["prompt_length"] != cand_request["prompt_length"]
            or ref_request["output_steps"] != cand_request["output_steps"]
        ):
            raise ComparisonError("control and candidate request work differs")
        tokens_equal = ref_request["output_tokens"] == cand_request["output_tokens"]
        all_tokens_equal = all_tokens_equal and tokens_equal
        ref_boundaries = boundary_map(ref_request)
        cand_boundaries = boundary_map(cand_request)
        if set(ref_boundaries) != set(cand_boundaries):
            raise ComparisonError("control and candidate boundaries differ")
        boundary_results = []
        for key in sorted(ref_boundaries):
            ref_values = read_activation(ref_boundaries[key])
            cand_values = read_activation(cand_boundaries[key])
            terms = vector_terms(ref_values, cand_values)
            total_diff2 += terms["diff2"]
            total_ref2 += terms["reference2"]
            total_cand2 += terms["candidate2"]
            total_dot += terms["dot"]
            maximum_abs = max(maximum_abs, terms["max_abs"])
            boundary_results.append({
                "layer_end": key[0],
                "position": key[1],
                "elements": len(ref_values),
                "reference_worker": ref_boundaries[key]["worker"],
                "candidate_worker": cand_boundaries[key]["worker"],
                "reference_sha256": ref_boundaries[key]["sha256"],
                "candidate_sha256": cand_boundaries[key]["sha256"],
                "relative_l2": terms["relative_l2"],
                "cosine": terms["cosine"],
                "max_abs": terms["max_abs"],
                "within_relative_l2_limit": (
                    terms["relative_l2"] <= relative_l2_limit
                ),
            })
        request_results.append({
            "request_id": request_id,
            "reference_route": ref_request["route_id"],
            "candidate_route": cand_request["route_id"],
            "tokens_equal": tokens_equal,
            "reference_tokens": ref_request["output_tokens"],
            "candidate_tokens": cand_request["output_tokens"],
            "boundaries": boundary_results,
        })

    aggregate_rel_l2 = math.sqrt(total_diff2 / total_ref2) if total_ref2 > 0 else (
        0.0 if total_diff2 == 0 else math.inf
    )
    aggregate_cosine = (
        total_dot / math.sqrt(total_ref2 * total_cand2)
        if total_ref2 > 0 and total_cand2 > 0
        else (1.0 if total_diff2 == 0 else 0.0)
    )
    boundaries_within = all(
        boundary["within_relative_l2_limit"]
        for request in request_results
        for boundary in request["boundaries"]
    )
    return {
        "schema": SCHEMA,
        "label": label,
        "scope": "SYNTHETIC_TOKEN_BOUNDARY_AND_GREEDY_OUTPUT_SCREEN",
        "reference": {
            "path": str(reference_path),
            "sha256": "sha256:" + sha256_file(reference_path),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": "sha256:" + sha256_file(candidate_path),
        },
        "request_count": len(request_results),
        "relative_l2_limit": relative_l2_limit,
        "aggregate": {
            "relative_l2": aggregate_rel_l2,
            "cosine": aggregate_cosine,
            "max_abs": maximum_abs,
            "all_tokens_equal": all_tokens_equal,
            "all_boundaries_within_limit": boundaries_within,
        },
        "requests": request_results,
        "boundary_gate_pass": boundaries_within,
        "token_screen_pass": all_tokens_equal,
        "quality_certified": False,
        "quality_note": (
            "This synthetic boundary screen is not a real-prompt output-quality "
            "certificate. The F16-phone/Q8-server route remains numerically "
            "uncertified unless the separate quality gate is added and passes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--relative-l2-limit", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = compare_runs(
            args.reference,
            args.candidate,
            args.label,
            args.relative_l2_limit,
        )
    except (ComparisonError, OSError, ValueError) as exc:
        print(json.dumps({
            "status": "FAIL", "error": str(exc),
        }, sort_keys=True, separators=(",", ":")))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report))
    print(json.dumps({
        "label": report["label"],
        "boundary_gate_pass": report["boundary_gate_pass"],
        "token_screen_pass": report["token_screen_pass"],
        "quality_certified": report["quality_certified"],
        "output": str(args.output),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
