#!/usr/bin/env python3
"""Classify normalized trace rows against the exact S11 measured envelope."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from s12lib import (
    MAX_SAFE_INT,
    S12Error,
    canonical_json,
    is_int,
    load_json,
    read_jsonl_snapshot,
    require_exact_keys,
    require_int,
    require_str,
    sha256_object,
    validate_profile,
    write_canonical,
)


REQUEST_KEYS = {
    "audio_ms",
    "cache_keys",
    "deadline_provenance",
    "deadline_us",
    "event_id",
    "images",
    "input_tokens",
    "model_class",
    "observed_latency_us",
    "output_tokens",
    "priority_class",
    "priority_provenance",
    "provenance",
    "retrieved_chunks",
    "schema_version",
    "service",
    "session_id",
    "source",
    "source_fields",
    "t_us",
}

MODES = {"strict_real", "semi_synthetic_shape_shadow"}
SYNTHETIC_SCOPE = "SYNTHETIC_ONLY_NO_REAL_TRACE_PERFORMANCE_CLAIM"
STRICT_SCOPE = "REAL_ARRIVALS_PROFILE_COVERAGE_ONLY"
FIXTURE_SCOPE = "SYNTHETIC_FIXTURE_PROFILE_COVERAGE_ONLY"


def validate_request(record: Any, index: int) -> dict[str, Any]:
    req = require_exact_keys(f"request[{index}]", record, REQUEST_KEYS)
    require_int(f"request[{index}].schema_version", req["schema_version"], 1, 1)
    for key in ("event_id", "source", "service", "model_class"):
        require_str(f"request[{index}].{key}", req[key])
    require_str(
        f"request[{index}].provenance",
        req["provenance"],
        {"real", "real_decomposed", "semi_synthetic", "synthetic"},
    )
    for key in ("t_us", "input_tokens", "output_tokens", "images", "audio_ms", "retrieved_chunks"):
        require_int(f"request[{index}].{key}", req[key], 0, MAX_SAFE_INT)
    if req["session_id"] is not None and type(req["session_id"]) is not str:
        raise S12Error(f"request[{index}].session_id: expected string or null")
    if req["observed_latency_us"] is not None:
        require_int(f"request[{index}].observed_latency_us", req["observed_latency_us"])
    if type(req["cache_keys"]) is not list or any(type(v) is not str for v in req["cache_keys"]):
        raise S12Error(f"request[{index}].cache_keys: expected string array")
    if type(req["source_fields"]) is not dict:
        raise S12Error(f"request[{index}].source_fields: expected object")
    for key, value in req["source_fields"].items():
        if type(key) is not str or type(value) not in (str, int, bool, type(None)):
            raise S12Error(f"request[{index}].source_fields: only scalar values are allowed")
    for value_key, provenance_key in (
        ("deadline_us", "deadline_provenance"),
        ("priority_class", "priority_provenance"),
    ):
        provenance = req[provenance_key]
        require_str(
            f"request[{index}].{provenance_key}",
            provenance,
            {"none", "synthetic"},
        )
        value = req[value_key]
        if provenance == "none" and value is not None:
            raise S12Error(f"request[{index}].{value_key}: must be null when provenance is none")
        if provenance == "synthetic" and value is None:
            raise S12Error(f"request[{index}].{value_key}: required when provenance is synthetic")
    if req["deadline_us"] is not None:
        require_int(f"request[{index}].deadline_us", req["deadline_us"])
    if req["priority_class"] is not None and (
        type(req["priority_class"]) is not str or not req["priority_class"]
    ):
        raise S12Error(f"request[{index}].priority_class: expected non-empty string or null")
    return req


def exact_match_reasons(req: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    model = profile["model"]
    reasons = []
    if req["provenance"] not in ("real", "real_decomposed"):
        reasons.append("E_PROVENANCE_NOT_REAL")
    checks = (
        ("INPUT_TOKENS", req["input_tokens"], model["prompt_tokens"]),
        ("OUTPUT_TOKENS", req["output_tokens"], model["generated_tokens"]),
        ("IMAGES", req["images"], 0),
        ("AUDIO_MS", req["audio_ms"], 0),
        ("RETRIEVED_CHUNKS", req["retrieved_chunks"], 0),
    )
    for name, actual, expected in checks:
        if actual != expected:
            reasons.append(f"E_{name}")
    source_fields = req["source_fields"]
    string_checks = (
        ("MODEL_ID", source_fields.get("s11_model_id"), model["model_id"]),
        (
            "HOST_MODEL_SHA256",
            source_fields.get("s11_host_model_sha256"),
            model["host_model_sha256"],
        ),
        ("PROMPT_SHA256", source_fields.get("s11_prompt_sha256"), model["prompt_sha256"]),
    )
    for name, actual, expected in string_checks:
        if type(actual) is not str or actual != expected:
            reasons.append(f"E_{name}")
    context = source_fields.get("s11_context_tokens")
    if not is_int(context) or context != model["context_tokens"]:
        reasons.append("E_CONTEXT_TOKENS")
    chat = source_fields.get("s11_chat")
    if type(chat) is not bool or chat is not True:
        reasons.append("E_CHAT")
    return reasons


def prepare_trace(
    records: list[Any],
    profile: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    require_str("mode", mode, MODES)
    if len(records) > 10000:
        raise S12Error("trace: exceeds 10000-record mechanics bound")
    requests = [validate_request(record, index) for index, record in enumerate(records)]
    order = [(req["t_us"], req["event_id"]) for req in requests]
    if order != sorted(order):
        raise S12Error("trace: rows must already be ordered by (t_us,event_id)")
    ids = [req["event_id"] for req in requests]
    if len(ids) != len(set(ids)):
        raise S12Error("trace: duplicate event_id")

    prepared = []
    eligible = 0
    for req in requests:
        reasons = exact_match_reasons(req, profile)
        if mode == "strict_real":
            is_eligible = not reasons
            if is_eligible:
                eligible += 1
            prepared.append(
                {
                    "arrival_us": req["t_us"],
                    "claim_scope": STRICT_SCOPE,
                    "event_id": req["event_id"],
                    "model_class": req["model_class"],
                    "original_input_tokens": req["input_tokens"],
                    "original_output_tokens": req["output_tokens"],
                    "profile_eligible": is_eligible,
                    "profile_key": "S11_GEMMA4_OP15_EXACT" if is_eligible else None,
                    "provenance": req["provenance"],
                    "reasons": reasons,
                    "service": req["service"],
                    "source": req["source"],
                }
            )
        else:
            eligible += 1
            prepared.append(
                {
                    "arrival_us": req["t_us"],
                    "claim_scope": SYNTHETIC_SCOPE,
                    "event_id": req["event_id"],
                    "model_class": profile["model"]["model_id"],
                    "original_input_tokens": req["input_tokens"],
                    "original_output_tokens": req["output_tokens"],
                    "profile_eligible": True,
                    "profile_key": "S11_GEMMA4_OP15_EXACT",
                    "provenance": "semi_synthetic",
                    "reasons": ["SHAPE_AND_PAYLOAD_REPLACED_WITH_S11_PROFILE"],
                    "service": req["service"],
                    "source": req["source"],
                }
            )

    if mode == "strict_real":
        scope = (
            STRICT_SCOPE
            if all(req["provenance"] in ("real", "real_decomposed") for req in requests)
            else FIXTURE_SCOPE
        )
        for request in prepared:
            request["claim_scope"] = scope
    else:
        scope = SYNTHETIC_SCOPE
    result = {
        "claim_scope": scope,
        "energy_status": "NOT_RUN",
        "mode": mode,
        "prepared_requests": prepared,
        "profile_digest": sha256_object(profile),
        "profiled_requests": eligible,
        "schema": "s12-profile-coverage-v1",
        "total_requests": len(prepared),
        "unprofiled_requests": len(prepared) - eligible,
    }
    result["coverage_digest"] = sha256_object(result)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--out")
    parser.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="repository root used only for --verify-artifacts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        profile_path = Path(args.profile)
        profile = validate_profile(
            load_json(profile_path),
            base_dir=args.repo_root,
            verify_artifacts=args.verify_artifacts,
        )
        trace_records, trace_digest = read_jsonl_snapshot(args.trace)
        result = prepare_trace(trace_records, profile, args.mode)
        result["profile_path"] = str(profile_path)
        result["trace_path"] = str(Path(args.trace))
        result["trace_sha256"] = trace_digest
        if args.out:
            write_canonical(args.out, result)
        else:
            print(canonical_json(result))
        return 0
    except S12Error as exc:
        print(f"PROFILE_COVERAGE_FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
