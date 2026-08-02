#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "s38-all-local-rag-v2"
COMPARISON_SCHEMA = "s38-matched-server-comparison-v1"


class ComparisonError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"), object_pairs_hook=reject_duplicate_keys)
    if not isinstance(value, dict):
        raise ComparisonError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line, object_pairs_hook=reject_duplicate_keys)
            if not isinstance(value, dict):
                raise ComparisonError(f"expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ComparisonError(f"empty request result: {path}")
    return rows


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def check_metric(summary: dict[str, Any], name: str, rows: list[dict[str, Any]], field: str) -> None:
    observed = summary.get(name)
    if not isinstance(observed, dict):
        raise ComparisonError(f"summary lacks {name}")
    values = [row[field] / 1000 for row in rows]
    expected = {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }
    if observed != expected:
        raise ComparisonError(f"summary {name} does not match request records")


def validate_result(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(directory / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ComparisonError(f"unexpected manifest schema in {directory}")
    for key in ("requests", "summary"):
        binding = manifest.get(key)
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ComparisonError(f"invalid {key} binding in {directory}")
        artifact = directory / binding["path"]
        if not artifact.is_file() or file_sha256(artifact) != binding["sha256"]:
            raise ComparisonError(f"invalid {key} artifact in {directory}")

    summary = load_json(directory / manifest["summary"]["path"])
    rows = load_jsonl(directory / manifest["requests"]["path"])
    if summary.get("schema") != SCHEMA or summary.get("status") != "S38_ALL_LOCAL_BASELINE_COMPLETE":
        raise ComparisonError(f"incomplete result in {directory}")
    if type(summary.get("requests")) is not int or summary["requests"] != len(rows):
        raise ComparisonError(f"request count mismatch in {directory}")

    seen: set[str] = set()
    integer_fields = (
        "arrival_due_us",
        "worker_start_us",
        "finish_us",
        "client_queue_us",
        "service_us",
        "response_us",
        "requested_input_tokens",
        "requested_output_tokens",
        "realized_prompt_tokens",
        "realized_output_tokens",
    )
    for row in rows:
        event_id = row.get("event_id")
        if row.get("schema") != SCHEMA or not isinstance(event_id, str) or event_id in seen:
            raise ComparisonError(f"invalid or duplicate request record in {directory}")
        seen.add(event_id)
        if any(type(row.get(field)) is not int or row[field] < 0 for field in integer_fields):
            raise ComparisonError(f"invalid integer field for {event_id}")
        arrival = row["arrival_due_us"]
        start = row["worker_start_us"]
        finish = row["finish_us"]
        if not arrival <= start <= finish:
            raise ComparisonError(f"invalid timeline for {event_id}")
        if row["client_queue_us"] != start - arrival:
            raise ComparisonError(f"queue mismatch for {event_id}")
        if row["response_us"] != finish - arrival:
            raise ComparisonError(f"response mismatch for {event_id}")
        if abs(row["service_us"] - (finish - start)) > 1:
            raise ComparisonError(f"service mismatch for {event_id}")
        stages = row.get("stages_us")
        if not isinstance(stages, dict) or any(type(value) is not int or value < 0 for value in stages.values()):
            raise ComparisonError(f"invalid stage timing for {event_id}")
        if abs(sum(stages.values()) - row["service_us"]) > len(stages):
            raise ComparisonError(f"stage timing mismatch for {event_id}")

    check_metric(summary, "client_queue_ms", rows, "client_queue_us")
    check_metric(summary, "service_ms", rows, "service_us")
    check_metric(summary, "response_ms", rows, "response_us")
    for summary_key, row_key in (
        ("requested_input_tokens", "requested_input_tokens"),
        ("requested_output_tokens", "requested_output_tokens"),
        ("realized_prompt_tokens", "realized_prompt_tokens"),
        ("realized_output_tokens", "realized_output_tokens"),
    ):
        if summary.get(summary_key) != sum(row[row_key] for row in rows):
            raise ComparisonError(f"summary {summary_key} mismatch in {directory}")
    return summary, rows


def compare(
    left_summary: dict[str, Any],
    left_rows: list[dict[str, Any]],
    right_summary: dict[str, Any],
    right_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    for key in ("models", "inputs", "arrival_mode", "workers", "retrieval_top_k", "rerank_top_n"):
        if left_summary.get(key) != right_summary.get(key):
            raise ComparisonError(f"matched-run field differs: {key}")
    left_by_id = {row["event_id"]: row for row in left_rows}
    right_by_id = {row["event_id"]: row for row in right_rows}
    if set(left_by_id) != set(right_by_id):
        raise ComparisonError("request event sets differ")
    identity_fields = (
        "payload_id",
        "arrival_due_us",
        "requested_input_tokens",
        "requested_output_tokens",
        "reasoning_budget_tokens",
    )
    event_ids = sorted(left_by_id)
    for event_id in event_ids:
        if any(left_by_id[event_id].get(key) != right_by_id[event_id].get(key) for key in identity_fields):
            raise ComparisonError(f"request identity differs: {event_id}")

    request_count = len(event_ids)
    retrieval_agreement = sum(
        left_by_id[event_id]["retrieved_chunk_ids"] == right_by_id[event_id]["retrieved_chunk_ids"]
        for event_id in event_ids
    ) / request_count
    retrieval_set_agreement = sum(
        set(left_by_id[event_id]["retrieved_chunk_ids"]) == set(right_by_id[event_id]["retrieved_chunk_ids"])
        for event_id in event_ids
    ) / request_count
    rerank_agreement = sum(
        left_by_id[event_id]["reranked_chunk_ids"] == right_by_id[event_id]["reranked_chunk_ids"]
        for event_id in event_ids
    ) / request_count
    rerank_set_agreement = sum(
        set(left_by_id[event_id]["reranked_chunk_ids"]) == set(right_by_id[event_id]["reranked_chunk_ids"])
        for event_id in event_ids
    ) / request_count
    answer_agreement = sum(
        left_by_id[event_id]["generated_answer"] == right_by_id[event_id]["generated_answer"]
        for event_id in event_ids
    ) / request_count
    normalized_answer_agreement = sum(
        re.findall(r"[a-z0-9]+", left_by_id[event_id]["generated_answer"].lower())
        == re.findall(r"[a-z0-9]+", right_by_id[event_id]["generated_answer"].lower())
        for event_id in event_ids
    ) / request_count

    return {
        "schema": COMPARISON_SCHEMA,
        "status": "S38_MATCHED_SERVER_BASELINES_VALID",
        "requests": request_count,
        "bindings_match": True,
        "left": left_summary,
        "right": right_summary,
        "relative": {
            "left_over_right_requests_per_s": left_summary["requests_per_s"] / right_summary["requests_per_s"],
            "left_over_right_output_tokens_per_s": (
                left_summary["realized_output_tokens_per_s"] / right_summary["realized_output_tokens_per_s"]
            ),
            "retrieved_chunk_order_agreement": retrieval_agreement,
            "retrieved_chunk_set_agreement": retrieval_set_agreement,
            "reranked_chunk_order_agreement": rerank_agreement,
            "reranked_chunk_set_agreement": rerank_set_agreement,
            "generated_answer_exact_agreement": answer_agreement,
            "generated_answer_normalized_agreement": normalized_answer_agreement,
        },
    }


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and compare two S38 all-local baselines")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left_summary, left_rows = validate_result(args.left.resolve())
    right_summary, right_rows = validate_result(args.right.resolve())
    result = compare(left_summary, left_rows, right_summary, right_rows)
    result["artifacts"] = {
        "left": {
            "directory": str(args.left.resolve()),
            "manifest_sha256": file_sha256(args.left.resolve() / "manifest.json"),
        },
        "right": {
            "directory": str(args.right.resolve()),
            "manifest_sha256": file_sha256(args.right.resolve() / "manifest.json"),
        },
    }
    encoded = canonical_json(result) + "\n"
    if args.output:
        atomic_text(args.output.resolve(), encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ComparisonError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"S38_COMPARISON_ERROR: {error}") from None
