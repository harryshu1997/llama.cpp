#!/usr/bin/env python3
"""Build a single-user multi-turn CONVERSATION replay trace from a WildChat sample (WS24, 2026-10-05).

Why: the BurstGPT traces replay an API arrival log with synthesized prompt bodies. A local personal helper
serves one user who holds conversations: every turn is mostly decode over a growing prefix, with think-time
gaps between turns. That is the workload the phone-helper design is for, so this builder makes such a trace
from real conversations (allenai/WildChat-1M, fetched by tools/fetch_wildchat_sample.py).

What a row is: one user turn. Its prompt is the chat template over the whole history (user_1, the dataset's
assistant_1, ..., user_k, assistant opener), tokenized SEGMENT by segment so turn k's prompt is the exact
token prefix of turn k+1's: the server's prefix cache reuses everything but the previous reply and the new
user message (that needs the campaign's opt-in ``cache_idle_slots`` launch parameter). The dataset's assistant
text stands in the history, so the trace is identical on every arm whatever the server generates.
``output_tokens`` is the dataset reply's token count, clipped to --output-cap.

Pacing (open loop, sequential by construction): one conversation at a time;
arrival(k+1) = arrival(k) + new_prefill_tokens x --prefill-s-per-token + output_tokens x --decode-s-per-token
+ think(k+1), think from WildChat's assistant timestamps (gap minus an estimated generation time, clipped to
[--think-min, --think-max], default --think-default when stamps are missing); --session-gap-s between
conversations. The service constants are desktop-only worst cases so turns never overlap on any arm; faster
arms simply idle longer, which is the single-user reality. Nothing here is a measurement.

Outputs (the campaign loader's four files + <trace>.md): REQUESTS_SEMANTIC_SOURCE.jsonl, REQUESTS_OVERLAY.jsonl
(>= 1 shortest single-turn message on the Llama overlay model), TRACE_MANIFEST.json, <trace>.json.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .build_realistic_trace import (OVERLAY_MODEL_ID, QWEN_TOKENIZER_MODEL, GEMMA_TOKENIZER_MODEL, Codec,
                                    digest_file, inventory_from_manifest, model_inventory, parse_execution_artifact)
from .trace_inputs import REPLAY_START_US
from .ws24_trace_common import (COLD_ROLE, HOT_ROLE, TEMPLATES, Pacing, TokenCodec, count_tokens, large_row,
                                overlay_row, sanitize, summary, think_time_s, tokenize_segment, write_trace)

SOURCE_DATASET = "allenai/WildChat-1M"


@dataclass(frozen=True)
class ConversationOptions:
    trace_name: str = "wildchat_conversation_v1"
    duration_s: float = 1800.0
    hot_share: float = 0.7
    context_cap: int = 6000
    output_cap: int = 512
    min_output: int = 4
    min_turns: int = 3
    session_gap_s: float = 90.0
    think_min_s: float = 15.0
    think_max_s: float = 300.0
    think_default_s: float = 90.0
    generation_chars_per_s: float = 60.0
    overlay_count: int = 1
    seed: int = 20261005
    pacing: Pacing = field(default_factory=Pacing)

    def to_json(self) -> dict[str, Any]:
        return {"trace_name": self.trace_name, "duration_s": self.duration_s, "hot_share": self.hot_share,
                "context_cap": self.context_cap, "output_cap": self.output_cap, "min_output": self.min_output,
                "min_turns": self.min_turns, "session_gap_s": self.session_gap_s,
                "think": {"minimum_s": self.think_min_s, "maximum_s": self.think_max_s, "default_s": self.think_default_s,
                          "generation_chars_per_s": self.generation_chars_per_s,
                          "rule": "assistant stamp gap minus next reply chars / generation_chars_per_s, clipped"},
                "overlay_count": self.overlay_count, "seed": self.seed, "pacing": self.pacing.to_json()}


def conversation_pairs(conversation: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(user, assistant) message pairs while the transcript alternates user/assistant with non-empty text."""
    pairs = []
    for index in range(0, len(conversation) - 1, 2):
        user, assistant = conversation[index], conversation[index + 1]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            break
        if not sanitize(str(user.get("content") or "")).strip() or not sanitize(str(assistant.get("content") or "")).strip():
            break
        pairs.append((user, assistant))
    return pairs


def eligible_conversations(sample: list[dict[str, Any]], options: ConversationOptions) -> list[dict[str, Any]]:
    rows = []
    for row in sample:
        if row.get("toxic") or row.get("redacted"):
            continue
        if row.get("language") not in (None, "English"):
            continue
        pairs = conversation_pairs(row.get("conversation") or [])
        if len(pairs) < options.min_turns:
            continue
        rows.append(row)
    rows.sort(key=lambda row: str(row.get("conversation_hash")))
    return rows


def role_for(index: int, hot_so_far: int, hot_share: float) -> str:
    """Running assignment: the first conversation is hot, the second cold, then the share is tracked."""
    if index == 0:
        return HOT_ROLE
    if index == 1:
        return COLD_ROLE
    return HOT_ROLE if hot_so_far / index < hot_share else COLD_ROLE


def build_conversation_trace(sample: list[dict[str, Any]], codecs: dict[str, TokenCodec], options: ConversationOptions,
                             *, llama_bytes: int, llama_sha: str) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Pure builder: (large_rows, overlay_rows, statistics). Deterministic for a given sample, codecs and options."""
    rng = random.Random(options.seed)
    pool = eligible_conversations(sample, options)
    if len(pool) < 2:
        raise SystemExit(f"need at least two eligible conversations (>= {options.min_turns} turns, clean, English); found {len(pool)}")
    rng.shuffle(pool)
    end_s = REPLAY_START_US / 1e6 + options.duration_s
    t = REPLAY_START_US / 1e6
    large: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    conversation_ends: list[float] = []
    stop_reason = "pool exhausted"
    hot_so_far = 0
    for conversation_index, row in enumerate(pool):
        role = role_for(conversation_index, hot_so_far, options.hot_share)
        template, codec = TEMPLATES[role], codecs[role]
        pairs = conversation_pairs(row["conversation"])
        prompt: list[int] = []
        previous_assistant = ""
        turn_rows: list[dict[str, Any]] = []
        truncated = None
        completion = t
        for turn_index, (user, assistant) in enumerate(pairs):
            user_text = sanitize(str(user["content"])).strip()
            assistant_text = sanitize(str(assistant["content"])).strip()
            segment = (template.first_segment(user_text) if turn_index == 0
                       else template.continuation_segment(previous_assistant, user_text))
            segment_tokens = tokenize_segment(codec, template, segment, first=turn_index == 0)
            candidate = prompt + segment_tokens
            source_output = count_tokens(codec, template, assistant_text)
            output_tokens = max(options.min_output, min(options.output_cap, source_output))
            if len(candidate) + output_tokens > options.context_cap:
                truncated = "context_cap"
                break
            if turn_index == 0:
                think, think_source = 0.0, "session start"
            else:
                think, think_source = think_time_s(pairs[turn_index - 1][1].get("timestamp"), assistant.get("timestamp"),
                                                   len(assistant_text), chars_per_s=options.generation_chars_per_s,
                                                   minimum_s=options.think_min_s, maximum_s=options.think_max_s,
                                                   default_s=options.think_default_s)
            arrival = completion + think if turn_index else t
            service = options.pacing.service_s(len(segment_tokens), output_tokens)
            if arrival + service > end_s:
                truncated = "duration"
                break
            codec.validate_detokenize(candidate)
            turn_rows.append(large_row(
                trace_name=options.trace_name, index=-1, template=template, arrival_us=int(round(arrival * 1e6)),
                prompt=candidate, output_tokens=output_tokens, extra={
                    "conversation_id": str(row.get("conversation_hash")), "turn_index": turn_index, "turns_total": -1,
                    "history_tokens": len(prompt), "new_tokens": len(segment_tokens),
                    "think_s": round(think, 3), "think_source": think_source, "service_estimate_s": round(service, 3),
                    "source_dataset": SOURCE_DATASET, "source_model": str(row.get("model")),
                    "source_output_tokens": source_output, "source_user_chars": len(user_text),
                    "source_assistant_chars": len(assistant_text)}))
            prompt, previous_assistant, completion = candidate, assistant_text, arrival + service
        if not turn_rows:
            stop_reason = f"conversation {conversation_index} did not fit ({truncated})"
            break
        for turn_row in turn_rows:
            turn_row["turns_total"] = len(turn_rows)
            turn_row["conversation_truncated_by"] = truncated
        large.extend(turn_rows)
        selected.append(row)
        conversation_ends.append(completion)
        hot_so_far += role == HOT_ROLE
        if truncated == "duration":
            stop_reason = "duration reached"
            break
        t = completion + options.session_gap_s
        if t >= end_s:
            stop_reason = "duration reached"
            break
    if len(selected) < 2:
        raise SystemExit("fewer than two conversations fit --duration-s; both roles need one (use a longer duration)")
    for index, turn_row in enumerate(large):
        turn_row["request_index"] = index
        turn_row["event_id"] = f"{options.trace_name}:{index:03d}"
    # Overlay: the shortest first user messages of conversations the trace does not use, one per session gap.
    chosen = {str(row.get("conversation_hash")) for row in selected}
    clean = [row for row in sample if not row.get("toxic") and not row.get("redacted")
             and row.get("language") in (None, "English") and conversation_pairs(row.get("conversation") or [])]
    clean.sort(key=lambda row: str(row.get("conversation_hash")))
    spare = [row for row in clean if str(row.get("conversation_hash")) not in chosen] or pool
    spare.sort(key=lambda row: (len(sanitize(str(row["conversation"][0]["content"])).strip()), str(row.get("conversation_hash"))))
    overlay: list[dict[str, Any]] = []
    template, codec = TEMPLATES["overlay"], codecs["overlay"]
    for overlay_index in range(max(1, min(options.overlay_count, 2))):
        row = spare[overlay_index % len(spare)]
        user, assistant = conversation_pairs(row["conversation"])[0]
        user_text = sanitize(str(user["content"])).strip()
        prompt = tokenize_segment(codec, template, template.first_segment(user_text), first=True)
        codec.validate_detokenize(prompt)
        source_output = count_tokens(codec, template, sanitize(str(assistant["content"])).strip())
        gap_after = conversation_ends[min(overlay_index, len(conversation_ends) - 1)] + 5.0 + 30.0 * (
            overlay_index if overlay_index >= len(conversation_ends) else 0)
        overlay.append(overlay_row(
            trace_name=options.trace_name, overlay_index=overlay_index, arrival_us=int(round(gap_after * 1e6)),
            prompt=prompt, output_tokens=max(options.min_output, min(options.output_cap, source_output)),
            llama_bytes=llama_bytes, llama_sha=llama_sha,
            extra={"conversation_id": str(row.get("conversation_hash")), "source_dataset": SOURCE_DATASET,
                   "source_model": str(row.get("model")), "source_output_tokens": source_output}))
    inputs = [float(r["input_tokens"]) for r in large]
    thinks = [float(r["think_s"]) for r in large if r["turn_index"] > 0]
    stats = {
        "conversations": len(selected), "turns": len(large),
        "roles": {HOT_ROLE: sum(r["model_id"] == TEMPLATES[HOT_ROLE].role_model_id for r in large),
                  COLD_ROLE: sum(r["model_id"] == TEMPLATES[COLD_ROLE].role_model_id for r in large)},
        "conversations_by_role": {HOT_ROLE: hot_so_far, COLD_ROLE: len(selected) - hot_so_far},
        "turns_per_conversation": summary([float(r["turns_total"]) for r in large if r["turn_index"] == 0]),
        "prompt_tokens": summary(inputs), "output_tokens": summary([float(r["output_tokens"]) for r in large]),
        "new_tokens": summary([float(r["new_tokens"]) for r in large]),
        "think_s": summary(thinks), "think_sources": {s: sum(1 for r in large if r["think_source"] == s) for s in
                                                      sorted({r["think_source"] for r in large})},
        "prefix_reusable_fraction": (sum(r["history_tokens"] for r in large) / sum(inputs)) if inputs else 0.0,
        "replay_span_s": (max(r["arrival_us"] for r in large + overlay) - REPLAY_START_US) / 1e6,
        "last_completion_s": conversation_ends[-1] - REPLAY_START_US / 1e6,
        "truncated": {reason: sum(1 for r in large if r["turn_index"] == 0 and r["conversation_truncated_by"] == reason)
                      for reason in ("context_cap", "duration")},
        "stop_reason": stop_reason, "overlay_rows": len(overlay),
        "selected_conversations": [str(row.get("conversation_hash")) for row in selected],
    }
    return large, overlay, stats


def markdown(options: ConversationOptions, stats: dict[str, Any], sample_path: Path, sample_sha: str) -> str:
    p = stats["prompt_tokens"]; o = stats["output_tokens"]; k = stats["think_s"]; n = stats["new_tokens"]
    return f"""# {options.trace_name}: one user's conversations replayed at think-time pace

Source: {SOURCE_DATASET} conversations (`{sample_path.name}`, sha256 `{sample_sha[:16]}...`), English, >= {options.min_turns}
turns, not toxic/redacted; seeded order (seed {options.seed}); one conversation at a time, {options.session_gap_s:.0f} s between
conversations; context cap {options.context_cap} tokens; output cap {options.output_cap} tokens. Window {options.duration_s:.0f} s.

| | Value |
| --- | ---: |
| Conversations / turns | {stats['conversations']} / {stats['turns']} ({stats['conversations_by_role'][HOT_ROLE]} hot (Qwen) / {stats['conversations_by_role'][COLD_ROLE]} cold (Gemma) conversations; {stats['roles'][HOT_ROLE]} / {stats['roles'][COLD_ROLE]} turns); {stats['overlay_rows']} overlay row(s) |
| Turns per conversation p50 / max | {stats['turns_per_conversation'].get('p50', 0):.0f} / {stats['turns_per_conversation'].get('max', 0):.0f} |
| Prompt (context) tokens p50 / p90 / max | {p.get('p50', 0):.0f} / {p.get('p90', 0):.0f} / {p.get('max', 0):.0f} |
| New prompt tokens per turn p50 / p90 | {n.get('p50', 0):.0f} / {n.get('p90', 0):.0f} |
| Output tokens p50 / p90 / max | {o.get('p50', 0):.0f} / {o.get('p90', 0):.0f} / {o.get('max', 0):.0f} |
| Think time p50 / p90 (s) | {k.get('p50', 0):.0f} / {k.get('p90', 0):.0f} ({stats['think_sources']}) |
| Prefix-reusable share of prompt tokens | {stats['prefix_reusable_fraction']:.1%} |
| Replay span / last estimated completion | {stats['replay_span_s']:.0f} s / {stats['last_completion_s']:.0f} s |
| Conversations cut by context cap / duration | {stats['truncated']['context_cap']} / {stats['truncated']['duration']} ({stats['stop_reason']}) |

Pacing: {options.pacing.to_json()['rule']} with prefill {options.pacing.prefill_s_per_token} s/token and decode
{options.pacing.decode_s_per_token} s/token (desktop-only worst-case ASSUMPTIONS, so turns never overlap on any arm);
think time = assistant-stamp gap minus reply chars / {options.generation_chars_per_s:.0f} chars/s, clipped to
[{options.think_min_s:.0f}, {options.think_max_s:.0f}] s, default {options.think_default_s:.0f} s without stamps.
Prompts are the real transcripts under each model's chat template, tokenized per segment so every turn's prompt is
the token-exact prefix of the next; the dataset's replies stand in the history (identical on every arm). The server
reuses the prefix only with the opt-in ``cache_idle_slots`` launch parameter. Built by `build_conversation_trace.py`;
nothing here is a measurement.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, required=True, help="WildChat sample JSON (list of rows, or {'rows': [...]})")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--trace-name", default="wildchat_conversation_v1")
    ap.add_argument("--codec", type=Path, required=True, help="llama-token-codec binary")
    ap.add_argument("--library-dir", type=Path, required=True)
    ap.add_argument("--qwen-tokenizer-model", type=Path, required=True)
    ap.add_argument("--gemma-tokenizer-model", type=Path, required=True)
    ap.add_argument("--llama-tokenizer-model", type=Path, required=True)
    ap.add_argument("--duration-s", type=float, default=1800.0)
    ap.add_argument("--hot-share", type=float, default=0.7)
    ap.add_argument("--context-cap", type=int, default=6000)
    ap.add_argument("--output-cap", type=int, default=512)
    ap.add_argument("--min-output", type=int, default=4)
    ap.add_argument("--min-turns", type=int, default=3)
    ap.add_argument("--session-gap-s", type=float, default=90.0)
    ap.add_argument("--think-min", type=float, default=15.0)
    ap.add_argument("--think-max", type=float, default=300.0)
    ap.add_argument("--think-default", type=float, default=90.0)
    ap.add_argument("--generation-chars-per-s", type=float, default=60.0)
    ap.add_argument("--prefill-s-per-token", type=float, default=0.02)
    ap.add_argument("--decode-s-per-token", type=float, default=0.65)
    ap.add_argument("--overlay-count", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261005)
    artifacts = ap.add_mutually_exclusive_group(required=True)
    artifacts.add_argument("--execution-artifact", type=parse_execution_artifact, action="append",
                           metavar="MODEL_ID=PATH[=KIND]")
    artifacts.add_argument("--model-inventory-from", type=Path, metavar="TRACE_MANIFEST")
    args = ap.parse_args(argv)
    inventory = (model_inventory(args.execution_artifact) if args.execution_artifact is not None
                 else inventory_from_manifest(args.model_inventory_from))
    if OVERLAY_MODEL_ID not in inventory or len(inventory) < 3:
        raise SystemExit("the model inventory must name the hot, cold and overlay execution models")
    # Overlay rows carry the EXECUTION artifact's identity (the inventory's), not the local tokenizer file's:
    # the tokenizer is the same for every quantization of the model, the pinned artifact is what the runner checks.
    overlay_identity = inventory[OVERLAY_MODEL_ID]
    options = ConversationOptions(
        trace_name=args.trace_name, duration_s=args.duration_s, hot_share=args.hot_share, context_cap=args.context_cap,
        output_cap=args.output_cap, min_output=args.min_output, min_turns=args.min_turns, session_gap_s=args.session_gap_s,
        think_min_s=args.think_min, think_max_s=args.think_max, think_default_s=args.think_default,
        generation_chars_per_s=args.generation_chars_per_s, overlay_count=args.overlay_count, seed=args.seed,
        pacing=Pacing(args.prefill_s_per_token, args.decode_s_per_token))
    loaded = json.loads(args.sample.read_text())
    sample = loaded["rows"] if isinstance(loaded, dict) else loaded
    qwen_sha, gemma_sha, llama_sha = (digest_file(p) for p in (args.qwen_tokenizer_model, args.gemma_tokenizer_model,
                                                                args.llama_tokenizer_model))
    codecs: dict[str, Codec] = {
        HOT_ROLE: Codec(args.codec, args.qwen_tokenizer_model, qwen_sha, args.library_dir),
        COLD_ROLE: Codec(args.codec, args.gemma_tokenizer_model, gemma_sha, args.library_dir),
        "overlay": Codec(args.codec, args.llama_tokenizer_model, llama_sha, args.library_dir)}
    try:
        large, overlay, stats = build_conversation_trace(sample, codecs, options, llama_bytes=overlay_identity["artifact_bytes"],
                                                         llama_sha=overlay_identity["artifact_sha256"])
    except BaseException:
        for codec in codecs.values():
            try:
                codec.close()
            except Exception:
                pass  # the builder's own error is the one to report
        raise
    for codec in codecs.values():
        codec.close()
    sample_sha = digest_file(args.sample)
    derivation = {
        "builder": "build_conversation_trace.py", "source_dataset": SOURCE_DATASET,
        "sample": str(args.sample), "sample_sha256": sample_sha, "options": options.to_json(),
        "statistics": {k: v for k, v in stats.items() if k != "selected_conversations"},
        "selected_conversations": stats["selected_conversations"],
        "tokenizers": {QWEN_TOKENIZER_MODEL: qwen_sha, GEMMA_TOKENIZER_MODEL: gemma_sha, OVERLAY_MODEL_ID: llama_sha},
        "overlay_tokenizer_file": {"path": str(args.llama_tokenizer_model), "sha256": llama_sha,
                                   "artifact_identity_in_rows": overlay_identity},
    }
    write_trace(args.output_dir, options.trace_name, large, overlay, inventory, derivation)
    (args.output_dir / f"{options.trace_name}.md").write_text(markdown(options, stats, args.sample, sample_sha))
    print(json.dumps({k: v for k, v in stats.items() if k != "selected_conversations"}, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
