#!/usr/bin/env python3
"""Build a deterministic, token-exact subset for S38 physical generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from compare_server_baselines import file_sha256, load_json, validate_result  # noqa: E402
from mixed_generation import LlamaServerTokenizer  # noqa: E402
from run_server_baseline import build_messages, http_json, load_jsonl  # noqa: E402


SCHEMA = "s38-physical-generation-subset-v1"


class SubsetError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def select_rows(
    rows: Sequence[dict[str, Any]],
    count: int,
    max_context: int,
    output_steps: int,
) -> list[dict[str, Any]]:
    if (
        type(count) is not int
        or count <= 0
        or type(max_context) is not int
        or max_context <= 0
        or type(output_steps) is not int
        or output_steps <= 0
    ):
        raise ValueError("invalid subset selection")
    eligible = []
    for row in rows:
        prompt = row.get("realized_prompt_tokens")
        event_id = row.get("event_id")
        if (
            type(prompt) is int
            and prompt > 0
            and prompt + output_steps <= max_context
            and type(row.get("requested_output_tokens")) is int
            and row["requested_output_tokens"] >= output_steps
            and isinstance(event_id, str)
            and event_id
        ):
            eligible.append(row)
    eligible.sort(key=lambda item: (item["realized_prompt_tokens"], item["event_id"]))
    if len(eligible) < count:
        raise SubsetError("not enough requests fit the physical context")
    if count == 1:
        return [eligible[len(eligible) // 2]]
    denominator = count - 1
    indexes = [
        (index * (len(eligible) - 1) + denominator // 2) // denominator
        for index in range(count)
    ]
    if len(set(indexes)) != count:
        raise SubsetError("selection produced duplicate rows")
    return [eligible[index] for index in indexes]


def parse_stop_tokens(value: str) -> tuple[int, ...]:
    try:
        tokens = tuple(int(item, 10) for item in value.split(","))
    except ValueError as error:
        raise SubsetError("stop tokens must be comma-separated integers") from error
    if not tokens or any(token < 0 for token in tokens) or len(set(tokens)) != len(tokens):
        raise SubsetError("stop tokens must be unique nonnegative integers")
    return tokens


def build_subset(args: argparse.Namespace) -> dict[str, Any]:
    baseline_dir = args.baseline_dir.resolve()
    payloads_path = args.payloads.resolve()
    index_dir = args.index_dir.resolve()
    model_path = args.model.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SubsetError(f"output already exists: {output}")
    if len(args.model_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.model_sha256
    ):
        raise SubsetError("invalid model SHA-256")
    if file_sha256(model_path) != args.model_sha256:
        raise SubsetError("physical model hash mismatch")

    summary, rows = validate_result(baseline_dir)
    payload_hash = file_sha256(payloads_path)
    if summary.get("inputs", {}).get("payloads_sha256") != payload_hash:
        raise SubsetError("payload artifact differs from the frozen baseline")
    index_manifest_path = index_dir / "manifest.json"
    index_manifest_hash = file_sha256(index_manifest_path)
    if summary.get("inputs", {}).get("index_manifest_sha256") != index_manifest_hash:
        raise SubsetError("index manifest differs from the frozen baseline")
    index_manifest = load_json(index_manifest_path)
    chunks_binding = index_manifest.get("outputs", {}).get("chunks", {})
    if set(chunks_binding) != {"path", "sha256"}:
        raise SubsetError("index manifest has an invalid chunks binding")
    chunks_path = index_dir / chunks_binding["path"]
    if file_sha256(chunks_path) != chunks_binding["sha256"]:
        raise SubsetError("chunk artifact differs from its manifest")

    _, payload_by_id = load_jsonl(payloads_path, "payload_id")
    _, chunk_by_id = load_jsonl(chunks_path, "chunk_id")
    selected = select_rows(rows, args.count, args.max_context, args.output_steps)
    tokenizer = LlamaServerTokenizer(args.tokenizer_url, http_json)
    stop_tokens = parse_stop_tokens(args.stop_tokens)

    requests = []
    for request_id, row in enumerate(selected, 1):
        payload_id = row.get("payload_id")
        payload = payload_by_id.get(payload_id)
        chunk_ids = row.get("reranked_chunk_ids")
        if payload is None or not isinstance(chunk_ids, list) or not chunk_ids:
            raise SubsetError(f"missing RAG inputs for {row.get('event_id')}")
        try:
            ranked = [chunk_by_id[chunk_id] for chunk_id in chunk_ids]
        except (KeyError, TypeError) as error:
            raise SubsetError(f"missing reranked chunk for {row.get('event_id')}") from error
        messages = build_messages(payload["query"], ranked)
        reasoning_budget = row.get("reasoning_budget_tokens")
        if type(reasoning_budget) is not int or reasoning_budget < 0:
            raise SubsetError(f"invalid reasoning budget for {row.get('event_id')}")
        prompt_tokens = tokenizer.encode_messages(
            messages,
            {"thinking_budget_tokens": reasoning_budget},
        )
        if len(prompt_tokens) != row["realized_prompt_tokens"]:
            raise SubsetError(
                f"prompt-token mismatch for {row['event_id']}: "
                f"{len(prompt_tokens)} != {row['realized_prompt_tokens']}"
            )
        if len(prompt_tokens) + args.output_steps > args.max_context:
            raise SubsetError(f"physical context overflow for {row['event_id']}")
        requests.append({
            "event_id": row["event_id"],
            "messages": messages,
            "output_steps": args.output_steps,
            "payload_id": payload_id,
            "prompt_token_count": len(prompt_tokens),
            "prompt_tokens": list(prompt_tokens),
            "reference_answer": payload.get("answer"),
            "reasoning_budget_tokens": reasoning_budget,
            "recorded_q8_answer": row.get("generated_answer"),
            "recorded_q8_finish_reason": row.get("generation_finish_reason"),
            "recorded_q8_output_tokens": row.get("realized_output_tokens"),
            "request_id": request_id,
            "stop_tokens": list(stop_tokens),
        })

    return {
        "model": {
            "file_type": args.file_type,
            "path": str(model_path),
            "sha256": args.model_sha256,
        },
        "protocol": {
            "count": args.count,
            "max_context": args.max_context,
            "output_steps": args.output_steps,
            "selection": "even_order_statistics_by_recorded_prompt_tokens_v1",
            "stop_tokens": list(stop_tokens),
            "tokenizer_url": args.tokenizer_url.rstrip("/"),
        },
        "requests": requests,
        "schema": SCHEMA,
        "sources": {
            "baseline_manifest_sha256": file_sha256(baseline_dir / "manifest.json"),
            "baseline_requests_sha256": file_sha256(baseline_dir / "requests.jsonl"),
            "index_manifest_sha256": index_manifest_hash,
            "payloads_sha256": payload_hash,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--file-type", type=int, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=3072)
    parser.add_argument("--output-steps", type=int, default=8)
    parser.add_argument("--stop-tokens", default="1,50,106")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if type(args.file_type) is not int or args.file_type < 0:
        raise SubsetError("file type must be nonnegative")
    subset = build_subset(args)
    content = canonical_json(subset) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(args.output.resolve(), content)
    digest = hashlib.sha256(content.encode("ascii")).hexdigest()
    print(canonical_json({
        "output": str(args.output.resolve()),
        "requests": len(subset["requests"]),
        "sha256": digest,
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, SubsetError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
