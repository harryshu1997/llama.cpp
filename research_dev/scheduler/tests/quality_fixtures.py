"""Synthetic GSM8K suite and fabricated campaign runs for the quality builder and scorer tests (no hardware)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from research_dev.scheduler.campaigns.burstgpt.quality import build_trace
from research_dev.scheduler.campaigns.burstgpt.quality.gsm8k import (
    EXPECTED_FIRST_TOKENS, EXPECTED_LAST_TOKENS, Gsm8kItem,
)

# the frozen longtail_eval_v2 model inventory (execution artifacts pinned by bytes and sha256)
INVENTORY = {
    "gemma-4-12b-q40-dequant-f16": {
        "artifact_bytes": 23832065056, "artifact_file": "gemma-4-12B-Q40-dequant-f16.gguf",
        "artifact_sha256": "ed76f2183d2d1d65091986033023e6c78d27f6276c1b0c5826cc92acf73538cf",
        "kind": "text_decoder_f16_proxy"},
    "llama-3.2-1b-instruct-q4_0": {
        "artifact_bytes": 770928288, "artifact_file": "Llama-3.2-1B-Instruct-Q4_0.gguf",
        "artifact_sha256": "4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad",
        "kind": "text_decoder"},
    "qwen3-14b-q4km-dequant-f16": {
        "artifact_bytes": 29543423360, "artifact_file": "Qwen3-14B-Q4KM-dequant-f16.gguf",
        "artifact_sha256": "d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718",
        "kind": "text_decoder_f16_proxy"},
}
MODEL_IDS = {"qwen": "qwen3-14b-q4km-dequant-f16", "gemma": "gemma-4-12b-q40-dequant-f16",
             "llama": "llama-3.2-1b-instruct-q4_0"}


class FakeCodec:
    """Deterministic stand-in for the token codec: template structure tokens around one token per character."""

    def __init__(self, role: str, broken: bool = False):
        self.role, self.broken, self.calls = role, broken, 0

    def tokenize(self, text: str) -> list[int]:
        self.calls += 1
        body = [1000 + ord(character) % 5000 for character in text]
        if self.broken:
            return body
        return list(EXPECTED_FIRST_TOKENS[self.role]) + body + list(EXPECTED_LAST_TOKENS[self.role])

    def validate_detokenize(self, tokens: list[int]) -> None:
        if not tokens:
            raise RuntimeError("empty")


def synthetic_items(count: int, split: str = "test") -> list[Gsm8kItem]:
    return [Gsm8kItem(split=split, index=index, question=f"Question {index}: what is {index} plus one?",
                      reference=Decimal(index + 1)) for index in range(count)]


def build_small_suite(directory: Path, *, shards: int = 2, per_shard=None, output_tokens: int = 12,
                      planned_shards: int = 1, items: int = 40, pilot: bool = False) -> dict:
    per_shard = per_shard or {"llama": 1, "gemma": 3, "qwen": 3}
    return build_trace.build_suite(
        items=synthetic_items(items, "train" if pilot else "test"), dataset={"name": "gsm8k", "split": "test"},
        output_dir=directory, trace_prefix="qpilot" if pilot else "qtest",
        codecs={role: FakeCodec(role) for role in build_trace.ROLES}, inventory=INVENTORY,
        seed=5, shards=shards, per_shard=per_shard,
        output_tokens={role: output_tokens for role in build_trace.ROLES}, planned_shards=planned_shards,
        tokenizers={}, pilot=pilot)


def key_rows(suite_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (suite_dir / "QUALITY_KEY.jsonl").read_text().splitlines()]


VOCABULARY: dict[str, int] = {}


def piece_ids(pieces: list[str]) -> list[int]:
    return [VOCABULARY.setdefault(piece, 5000 + len(VOCABULARY)) for piece in pieces]


def answer_pieces(value: str | None, budget: int, *, lead: str = "Step one", tail: str = " more",
                  position: int = 0) -> list[str]:
    """`budget` stream pieces that state 'Final answer: value' after `position` filler pieces and then keep
    going (EOS is banned); value None never states an answer."""
    if value is None:
        return [tail] * budget
    pieces = [" step"] * position + [lead, ".", "\n", "Final", " answer", ":", " " + value, "\n"]
    pieces += [tail] * max(0, budget - len(pieces))
    return pieces[:budget]


def write_stream(path: Path, pieces: list[str], *, chunk: int = 1) -> None:
    ids = piece_ids(pieces)
    lines = []
    for start in range(0, len(pieces), chunk):
        lines.append("data: " + json.dumps({"index": 0, "content": "".join(pieces[start:start + chunk]),
                                            "tokens": ids[start:start + chunk], "stop": False,
                                            "tokens_predicted": min(len(pieces), start + chunk)}) + "\n\n")
    lines.append("data: " + json.dumps({"index": 0, "content": "", "tokens": [], "stop": True,
                                        "tokens_predicted": len(pieces)}) + "\n\n")
    path.write_text("".join(lines))


def write_run(run_dir: Path, trace_name: str, rows: list[dict], outputs: dict[str, list[str] | None], *,
              coverage: dict[str, float] | None = None, rejected: tuple[str, ...] = (), chunk: int = 1,
              prompt_override: dict[str, str] | None = None) -> None:
    """RESULT.json + streams/ for one shard: outputs maps request_id -> pieces (None: no stream written)."""
    (run_dir / "streams").mkdir(parents=True)
    results = []
    for row in rows:
        if row["trace_name"] != trace_name or row["request_id"] in rejected:
            continue
        share = (coverage or {}).get(row["request_id"], 0.0)
        steps = row["output_tokens"] - 1
        calls = int(round(share * steps))
        results.append({
            "request_id": row["request_id"], "combined_request_index": row["combined_request_index"],
            "prompt_sha256": (prompt_override or {}).get(row["request_id"], row["prompt_sha256"]),
            "output_tokens": row["output_tokens"], "model_id": MODEL_IDS[row["role"]],
            "actual_executor_id": "physical:test:desktop",
            "physical_execution_proof": {"phone_call_count": calls * 2,
                                         "phone_calls_by_layer": [{"layer": 0, "calls": calls},
                                                                  {"layer": 1, "calls": calls}]},
        })
        pieces = outputs.get(row["request_id"])
        if pieces is not None:
            write_stream(run_dir / "streams" / f"request-{row['combined_request_index']:03d}.raw", pieces,
                         chunk=chunk)
    result = {"status": "PASS", "replay_schedule": {"trace_name": trace_name}, "request_results": results,
              "rejected_requests": [{"request_id": request_id} for request_id in rejected]}
    (run_dir / "RESULT.json").write_text(json.dumps(result))
