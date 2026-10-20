"""Shared pieces of the WS24 trace builders (conversation and long-prompt traces).

The builders in ``build_conversation_trace.py`` and ``build_longprompt_trace.py`` emit the same four
files the campaign loader consumes (see ``build_realistic_trace.py``); this module holds what they share:

* the chat templates of the hot (Qwen) and cold (Gemma) roles and the Llama overlay, as SEGMENTS, so a
  multi-turn prompt is the token-exact concatenation of its previous prompt and one new segment and the
  server's prefix cache reuses the earlier turns;
* the open-loop sequential pacing model (service estimate + think time) that keeps a single user's turns
  from overlapping even on the slowest arm;
* the think-time derivation from WildChat assistant timestamps;
* the writer of REQUESTS_SEMANTIC_SOURCE.jsonl / REQUESTS_OVERLAY.jsonl / TRACE_MANIFEST.json /
  <trace>.json, byte-compatible with the realistic builder's manifest.

Nothing here is a measurement; every constant below is an assumption and is recorded in the manifest.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .build_realistic_trace import (MANIFEST_SCHEMA, OVERLAY_MODEL_ID, OVERLAY_SCHEMA, OVERLAY_STREAM_ID,
                                    QWEN_TOKENIZER_MODEL, GEMMA_TOKENIZER_MODEL, SLO_US, SOURCE_SCHEMA,
                                    assign_combined_indices, canonical, digest_bytes)
from .trace_inputs import REPLAY_SCHEDULE_SCHEMA, REPLAY_START_US, TRACE_COLD_MODEL_ID, TRACE_HOT_MODEL_ID

HOT_ROLE, COLD_ROLE = "hot", "cold"

# Strings a user could paste that would otherwise be parsed as control tokens (the codec parses specials).
TEMPLATE_MARKERS = ("<|im_start|>", "<|im_end|>", "<|turn>", "<turn|>", "<|channel>", "<channel|>",
                    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|begin_of_text|>")


class TokenCodec(Protocol):
    def tokenize(self, text: str) -> list[int]: ...
    def validate_detokenize(self, tokens: list[int]) -> None: ...


@dataclass(frozen=True)
class ChatTemplate:
    """One model's chat markup, split so that every history step is its own tokenizable segment."""
    role: str                 # "hot" / "cold" / "overlay"
    role_model_id: str        # the trace row's model_id (role id for hot/cold, the overlay model id otherwise)
    tokenizer_model: str      # prompt_tokenizer_model
    user_open: str
    user_close: str           # follows the user text (starts with the newline the single-turn builders used)
    assistant_open: str       # the final opener the server generates after
    assistant_close: str      # closes a history assistant turn
    bos: int | None           # BOS the codec prepends to every tokenization (stripped from later segments)

    def first_segment(self, user_text: str) -> str:
        return self.user_open + user_text + self.user_close + self.assistant_open

    def continuation_segment(self, previous_assistant_text: str, user_text: str) -> str:
        return (previous_assistant_text + self.assistant_close + self.user_open + user_text + self.user_close
                + self.assistant_open)


# The single-turn builders' prefix/suffix pairs, decomposed. Qwen: "<|im_start|>user\n" ... "\n<|im_end|>\n<|im_start|>assistant\n".
# Gemma: "<|turn>user\n" ... "\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>" (BOS 2 from the codec).
TEMPLATES: dict[str, ChatTemplate] = {
    HOT_ROLE: ChatTemplate(HOT_ROLE, TRACE_HOT_MODEL_ID, QWEN_TOKENIZER_MODEL,
                           "<|im_start|>user\n", "\n<|im_end|>\n", "<|im_start|>assistant\n", "\n<|im_end|>\n", None),
    COLD_ROLE: ChatTemplate(COLD_ROLE, TRACE_COLD_MODEL_ID, GEMMA_TOKENIZER_MODEL,
                            "<|turn>user\n", "\n<turn|>\n", "<|turn>model\n<|channel>thought\n<channel|>",
                            "\n<turn|>\n", 2),
    "overlay": ChatTemplate("overlay", OVERLAY_MODEL_ID, OVERLAY_MODEL_ID,
                            "<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>",
                            "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>", 128000),
}


def sanitize(text: str) -> str:
    """Drop template control strings from user-provided text (they would tokenize as control tokens)."""
    for marker in TEMPLATE_MARKERS:
        text = text.replace(marker, "")
    return text


def tokenize_segment(codec: TokenCodec, template: ChatTemplate, text: str, *, first: bool) -> list[int]:
    """Tokenize one segment; a later segment loses the BOS the codec adds, so concatenations stay exact."""
    tokens = codec.tokenize(text)
    if not first and template.bos is not None and tokens and tokens[0] == template.bos:
        tokens = tokens[1:]
    return tokens


def count_tokens(codec: TokenCodec, template: ChatTemplate, text: str) -> int:
    """Token count of free text alone (no BOS), for output lengths."""
    if not text.strip():
        return 0
    return len(tokenize_segment(codec, template, text, first=False))


@dataclass(frozen=True)
class Pacing:
    """Open-loop, sequential pacing: the next request is scheduled after the previous one would have
    finished on the slowest arm plus the user's think time. Both numbers are ASSUMPTIONS, recorded in
    the manifest: prefill 0.02 s/token and decode 0.65 s/token are desktop-only Qwen worst cases."""
    prefill_s_per_token: float = 0.02
    decode_s_per_token: float = 0.65

    def service_s(self, new_prefill_tokens: int, output_tokens: int) -> float:
        return new_prefill_tokens * self.prefill_s_per_token + output_tokens * self.decode_s_per_token

    def to_json(self) -> dict[str, float]:
        return {"prefill_s_per_token": self.prefill_s_per_token, "decode_s_per_token": self.decode_s_per_token,
                "rule": "arrival(k+1) = arrival(k) + new_prefill_tokens x prefill + output_tokens x decode + think"}


def parse_timestamp(value: object) -> datetime | None:
    if type(value) is not str or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def think_time_s(previous_assistant_ts: object, next_assistant_ts: object, next_reply_chars: int, *,
                 chars_per_s: float = 60.0, minimum_s: float = 15.0, maximum_s: float = 300.0,
                 default_s: float = 90.0) -> tuple[float, str]:
    """User think time before the next turn, from WildChat's assistant timestamps.

    WildChat stamps assistant messages only. The gap between two assistant stamps covers the user's reading
    and typing plus the API's generation of the next reply; the generation is estimated at ``chars_per_s``
    (an ASSUMPTION) and subtracted. The result is clipped to [minimum_s, maximum_s]; without two stamps the
    default is used. Returns (seconds, source) with source "timestamps" or "default".
    """
    t1, t2 = parse_timestamp(previous_assistant_ts), parse_timestamp(next_assistant_ts)
    if t1 is None or t2 is None or t2 < t1:
        return default_s, "default"
    think = (t2 - t1).total_seconds() - next_reply_chars / chars_per_s
    return min(maximum_s, max(minimum_s, think)), "timestamps"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]


def summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "p50": statistics.median(values), "p90": percentile(values, 0.9),
            "max": max(values), "mean": statistics.fmean(values)}


def large_row(*, trace_name: str, index: int, template: ChatTemplate, arrival_us: int, prompt: list[int],
              output_tokens: int, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "arrival_us": arrival_us,
        "event_id": f"{trace_name}:{index:03d}",
        "input_tokens": len(prompt),
        "model_id": template.role_model_id,
        "output_tokens": output_tokens,
        "prompt_tokenizer_model": template.tokenizer_model,
        "prompt_tokens": prompt,
        "request_index": index,
        "schema": SOURCE_SCHEMA,
        "slo_us": SLO_US,
        **extra,
    }


def overlay_row(*, trace_name: str, overlay_index: int, arrival_us: int, prompt: list[int], output_tokens: int,
                llama_bytes: int, llama_sha: str, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "arrival_us": arrival_us,
        "combined_request_index": -1,  # assigned by write_trace once the merge order is known
        "event_id": f"{trace_name}:{OVERLAY_STREAM_ID}:{overlay_index:02d}",
        "execution_model_id": OVERLAY_MODEL_ID,
        "input_tokens": len(prompt),
        "modality": "text",
        "model_artifact_bytes": llama_bytes,
        "model_artifact_sha256": llama_sha,
        "model_id": OVERLAY_MODEL_ID,
        "output_tokens": output_tokens,
        "overlay_request_index": overlay_index,
        "prompt_tokenizer_model": OVERLAY_MODEL_ID,
        "prompt_tokens": prompt,
        "prompt_transport": "tokens",
        "requested_model_id": OVERLAY_MODEL_ID,
        "schema": OVERLAY_SCHEMA,
        "slo_us": SLO_US,
        "stream_request_index": overlay_index,
        "trace_stream_id": OVERLAY_STREAM_ID,
        **extra,
    }


def write_trace(out: Path, trace_name: str, large_rows: list[dict[str, Any]], overlay_rows: list[dict[str, Any]],
                inventory: dict[str, dict[str, Any]], derivation: dict[str, Any]) -> dict[str, Any]:
    """Write the four loader files; returns the manifest. Rows must already be in arrival order."""
    if not large_rows or not overlay_rows:
        raise SystemExit("a trace needs at least one large-model row and one overlay row")
    for name in ("REQUESTS_SEMANTIC_SOURCE.jsonl", "REQUESTS_OVERLAY.jsonl", "TRACE_MANIFEST.json", f"{trace_name}.json"):
        if (out / name).exists():
            raise SystemExit(f"{out / name} already exists; choose a fresh output directory")
    if [row["arrival_us"] for row in large_rows] != sorted(row["arrival_us"] for row in large_rows):
        raise SystemExit("large rows are not in arrival order")
    order = assign_combined_indices(large_rows, overlay_rows)
    for source_kind, source_index, combined in order:
        if source_kind == "overlay":
            overlay_rows[source_index]["combined_request_index"] = combined
    arrivals = []
    for source_kind, source_index, combined in order:
        row = large_rows[source_index] if source_kind == "large" else overlay_rows[source_index]
        arrivals.append({"combined_request_index": combined, "replay_arrival_us": row["arrival_us"]})
    source = b"".join(canonical(row) for row in large_rows)
    overlay = b"".join(canonical(row) for row in overlay_rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "REQUESTS_SEMANTIC_SOURCE.jsonl").write_bytes(source)
    (out / "REQUESTS_OVERLAY.jsonl").write_bytes(overlay)
    hot_count = sum(1 for row in large_rows if row["model_id"] == TRACE_HOT_MODEL_ID)
    cold_count = len(large_rows) - hot_count
    input_total = sum(row["input_tokens"] for row in large_rows)
    output_total = sum(row["output_tokens"] for row in large_rows)
    overlay_input = sum(row["input_tokens"] for row in overlay_rows)
    overlay_output = sum(row["output_tokens"] for row in overlay_rows)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "base_trace": {
            "path": str(out / "REQUESTS_SEMANTIC_SOURCE.jsonl"), "schema": SOURCE_SCHEMA,
            "sha256": digest_bytes(source), "record_count": len(large_rows),
            "roles": {HOT_ROLE: hot_count, COLD_ROLE: cold_count},
            "input_tokens": input_total, "output_tokens": output_total,
        },
        "overlay_trace": {
            "path": str(out / "REQUESTS_OVERLAY.jsonl"), "schema": OVERLAY_SCHEMA,
            "sha256": digest_bytes(overlay), "record_count": len(overlay_rows),
            "input_tokens": overlay_input, "output_tokens": overlay_output,
            "arrival_first_us": overlay_rows[0]["arrival_us"], "arrival_last_us": overlay_rows[-1]["arrival_us"],
            "stream_id": OVERLAY_STREAM_ID,
        },
        "combined_work": {"record_count": len(large_rows) + len(overlay_rows),
                          "input_tokens": input_total + overlay_input, "output_tokens": output_total + overlay_output},
        "execution_models": {HOT_ROLE: hot_count, COLD_ROLE: cold_count, OVERLAY_MODEL_ID: len(overlay_rows)},
        "model_inventory": inventory,
        "derivation": derivation,
    }
    (out / "TRACE_MANIFEST.json").write_bytes(canonical(manifest))
    schedule = {"arrivals": arrivals, "schema": REPLAY_SCHEDULE_SCHEMA, "trace_name": trace_name}
    (out / f"{trace_name}.json").write_bytes(canonical(schedule))
    return manifest


def role_sequence(count: int, hot_share: float, rng) -> list[str]:
    """A seeded role assignment with at least one conversation (or request) per role."""
    if count < 2:
        raise SystemExit("both roles need at least one conversation/request; the window holds fewer than two")
    hot_count = min(count - 1, max(1, round(count * hot_share)))
    roles = [HOT_ROLE] * hot_count + [COLD_ROLE] * (count - hot_count)
    rng.shuffle(roles)
    return roles


__all__ = ["COLD_ROLE", "HOT_ROLE", "ChatTemplate", "Pacing", "REPLAY_START_US", "TEMPLATES", "TokenCodec",
           "count_tokens", "large_row", "overlay_row", "parse_timestamp", "percentile", "role_sequence", "sanitize",
           "summary", "think_time_s", "tokenize_segment", "write_trace"]
