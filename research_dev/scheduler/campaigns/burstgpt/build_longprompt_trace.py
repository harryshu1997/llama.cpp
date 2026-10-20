#!/usr/bin/env python3
"""Build a single-user LONG-PROMPT replay trace from the Mooncake conversation trace (WS24, 2026-10-05).

Why: retrieval-augmented and document chat sends long prompts (thousands of tokens) for short answers. The
phone helpers serve decode, and prefill stays on the desktop, so this is the BOUNDARY case for the design: the
trace exists to measure how small (or negative) the saving gets, not to show a win. Mooncake's conversation
trace (kvcache-ai/Mooncake, FAST'25 release) has that shape: input p50 ~6.9k tokens, output p50 ~350.

Rows: N requests sampled (seeded) from the Mooncake rows with --min-input <= input_length <= --prompt-cap and
--min-output <= output_length <= --output-cap, paced sequentially like the conversation trace (one user; the
next request arrives after the previous one's estimated service plus --think-s). Two variants:

* ``fresh``  - every request carries its own synthesized document (the realistic builder's templated body cut
               to the source token count): no prefix reuse, the pure prefill-bound case.
* ``cached`` - every request of a role starts with the SAME synthesized document (--document-tokens tokens,
               token-exact) followed by a unique question, so the server's prefix cache applies after the
               first request of each role (needs the opt-in ``cache_idle_slots`` launch parameter). The row's
               ``history_tokens`` is the document length (what a warm cache reuses); ``new_tokens`` the rest.

Prompt token counts equal the Mooncake input_length except when the cached variant's question would be
shorter than --min-question-tokens (recorded in the row). Nothing here is a measurement.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .build_realistic_trace import (OVERLAY_MODEL_ID, QWEN_TOKENIZER_MODEL, GEMMA_TOKENIZER_MODEL, Codec,
                                    digest_file, inventory_from_manifest, model_inventory, parse_execution_artifact,
                                    prompt_body)
from .trace_inputs import REPLAY_START_US
from .ws24_trace_common import (COLD_ROLE, HOT_ROLE, TEMPLATES, ChatTemplate, Pacing, TokenCodec, large_row,
                                overlay_row, role_sequence, summary, tokenize_segment, write_trace)

SOURCE_DATASET = "kvcache-ai/Mooncake FAST25-release/traces/conversation_trace.jsonl"
VARIANTS = ("fresh", "cached")


@dataclass(frozen=True)
class LongPromptOptions:
    trace_name: str = "mooncake_longprompt_fresh_v1"
    variant: str = "fresh"
    requests: int = 12
    min_input: int = 2000
    prompt_cap: int = 8000
    min_output: int = 4
    output_cap: int = 512
    hot_share: float = 0.75
    think_s: float = 60.0
    document_tokens: int = 4000
    min_question_tokens: int = 64
    overlay_input_tokens: int = 48
    overlay_output_tokens: int = 32
    seed: int = 20261005
    pacing: Pacing = field(default_factory=Pacing)

    def to_json(self) -> dict[str, Any]:
        return {"trace_name": self.trace_name, "variant": self.variant, "requests": self.requests,
                "min_input": self.min_input, "prompt_cap": self.prompt_cap, "min_output": self.min_output,
                "output_cap": self.output_cap, "hot_share": self.hot_share, "think_s": self.think_s,
                "document_tokens": self.document_tokens, "min_question_tokens": self.min_question_tokens,
                "overlay": {"input_tokens": self.overlay_input_tokens, "output_tokens": self.overlay_output_tokens},
                "seed": self.seed, "pacing": self.pacing.to_json()}


def load_mooncake_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if type(row) is not dict or not all(type(row.get(k)) is int for k in ("timestamp", "input_length", "output_length")):
            raise SystemExit(f"{path}:{line_number}: not a Mooncake row")
        rows.append(row)
    rows.sort(key=lambda row: (row["timestamp"], row["input_length"], row["output_length"]))
    return rows


def exact_tokens(codec: TokenCodec, template: ChatTemplate, text: str, target: int, *, first: bool) -> list[int]:
    """Tokenize ``text`` and cut to exactly ``target`` tokens (the text must tokenize to at least that many)."""
    tokens = tokenize_segment(codec, template, text, first=first)
    if len(tokens) < target or target <= 0:
        raise SystemExit(f"synthesized text gives {len(tokens)} tokens, need {target}")
    return tokens[:target]


def document_text(role: str, tokens: int) -> str:
    return (f"Shared document for the {role} model. The following reference material is attached to every question. "
            + prompt_body(0, tokens))


def question_text(index: int, tokens: int) -> str:
    return f"Question {index}: using only the material above, " + prompt_body(index, tokens)


def build_longprompt_trace(rows: list[dict[str, Any]], codecs: dict[str, TokenCodec], options: LongPromptOptions, *,
                           llama_bytes: int, llama_sha: str) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Pure builder: (large_rows, overlay_rows, statistics)."""
    if options.variant not in VARIANTS:
        raise SystemExit(f"--variant must be one of {VARIANTS}")
    eligible = [r for r in rows if options.min_input <= r["input_length"] <= options.prompt_cap
                and options.min_output <= r["output_length"] <= options.output_cap]
    if len(eligible) < options.requests:
        raise SystemExit(f"only {len(eligible)} Mooncake rows fit the caps; --requests {options.requests}")
    rng = random.Random(options.seed)
    picked = sorted(rng.sample(range(len(eligible)), options.requests))
    sample = [eligible[i] for i in picked]
    roles = role_sequence(len(sample), options.hot_share, rng)
    shared: dict[str, list[int]] = {}
    if options.variant == "cached":
        for role in (HOT_ROLE, COLD_ROLE):
            template = TEMPLATES[role]
            shared[role] = exact_tokens(codecs[role], template, template.user_open + document_text(role, options.document_tokens),
                                        options.document_tokens, first=True)
    large: list[dict[str, Any]] = []
    t = REPLAY_START_US / 1e6
    seen_roles: set[str] = set()
    completions: list[float] = []
    for index, (row, role) in enumerate(zip(sample, roles)):
        template, codec = TEMPLATES[role], codecs[role]
        target = row["input_length"]
        output_tokens = row["output_length"]
        closing = template.user_close + template.assistant_open
        if options.variant == "fresh":
            suffix = tokenize_segment(codec, template, closing, first=False)
            body = exact_tokens(codec, template, template.user_open + prompt_body(index + 1, target), target - len(suffix), first=True)
            prompt = body + suffix
            history, question_short = 0, False
        else:
            question_target = target - options.document_tokens
            question_short = question_target < options.min_question_tokens
            question_target = max(question_target, options.min_question_tokens)
            suffix = tokenize_segment(codec, template, closing, first=False)
            question = exact_tokens(codec, template, "\n\n" + question_text(index + 1, question_target),
                                    question_target - len(suffix), first=False)
            prompt = shared[role] + question + suffix
            history = options.document_tokens if role in seen_roles else 0
        codec.validate_detokenize(prompt)
        new_tokens = len(prompt) - history
        service = options.pacing.service_s(new_tokens, output_tokens)
        arrival = t
        large.append(large_row(trace_name=options.trace_name, index=index, template=template, arrival_us=int(round(arrival * 1e6)),
                               prompt=prompt, output_tokens=output_tokens, extra={
                                   "variant": options.variant, "history_tokens": history, "new_tokens": new_tokens,
                                   "shared_document_tokens": options.document_tokens if options.variant == "cached" else 0,
                                   "question_padded_to_minimum": question_short,
                                   "think_s": options.think_s if index else 0.0, "service_estimate_s": round(service, 3),
                                   "source_dataset": SOURCE_DATASET, "source_timestamp_ms": row["timestamp"],
                                   "source_input_tokens": row["input_length"], "source_output_tokens": row["output_length"],
                                   "source_hash_ids": list(row.get("hash_ids", []))}))
        seen_roles.add(role)
        completions.append(arrival + service)
        t = arrival + service + options.think_s
    template, codec = TEMPLATES["overlay"], codecs["overlay"]
    suffix = tokenize_segment(codec, template, template.user_close + template.assistant_open, first=False)
    body = exact_tokens(codec, template, template.user_open + prompt_body(1, options.overlay_input_tokens),
                        options.overlay_input_tokens - len(suffix), first=True)
    overlay_prompt = body + suffix
    codec.validate_detokenize(overlay_prompt)
    overlay = [overlay_row(trace_name=options.trace_name, overlay_index=0, arrival_us=int(round((completions[0] + 5.0) * 1e6)),
                           prompt=overlay_prompt, output_tokens=options.overlay_output_tokens, llama_bytes=llama_bytes,
                           llama_sha=llama_sha, extra={"variant": options.variant, "source_dataset": "synthesized"})]
    inputs = [float(r["input_tokens"]) for r in large]
    stats = {
        "requests": len(large), "variant": options.variant,
        "roles": {HOT_ROLE: roles.count(HOT_ROLE), COLD_ROLE: roles.count(COLD_ROLE)},
        "prompt_tokens": summary(inputs), "new_tokens": summary([float(r["new_tokens"]) for r in large]),
        "output_tokens": summary([float(r["output_tokens"]) for r in large]),
        "prefix_reusable_fraction": sum(r["history_tokens"] for r in large) / sum(inputs) if inputs else 0.0,
        "questions_padded_to_minimum": sum(1 for r in large if r["question_padded_to_minimum"]),
        "prompt_tokens_differ_from_source": sum(1 for r in large if r["input_tokens"] != r["source_input_tokens"]),
        "think_s": options.think_s, "replay_span_s": (large[-1]["arrival_us"] - REPLAY_START_US) / 1e6,
        "last_completion_s": completions[-1] - REPLAY_START_US / 1e6, "eligible_source_rows": len(eligible),
        "overlay_rows": len(overlay),
    }
    return large, overlay, stats


def markdown(options: LongPromptOptions, stats: dict[str, Any], source: Path, source_sha: str) -> str:
    p = stats["prompt_tokens"]; o = stats["output_tokens"]; n = stats["new_tokens"]
    cached = options.variant == "cached"
    return f"""# {options.trace_name}: a single user's long-prompt requests ({options.variant} variant)

Source: {SOURCE_DATASET} (`{source.name}`, sha256 `{source_sha[:16]}...`): {stats['eligible_source_rows']} rows with
{options.min_input} <= input <= {options.prompt_cap} and {options.min_output} <= output <= {options.output_cap} tokens; {options.requests} sampled
(seed {options.seed}), paced one at a time with {options.think_s:.0f} s think time after each estimated completion.

| | Value |
| --- | ---: |
| Requests | {stats['requests']}: {stats['roles'][HOT_ROLE]} hot (Qwen) / {stats['roles'][COLD_ROLE]} cold (Gemma); {stats['overlay_rows']} overlay row |
| Prompt tokens p50 / p90 / max | {p.get('p50', 0):.0f} / {p.get('p90', 0):.0f} / {p.get('max', 0):.0f} |
| New (uncached) prompt tokens p50 / p90 | {n.get('p50', 0):.0f} / {n.get('p90', 0):.0f} |
| Output tokens p50 / p90 / max | {o.get('p50', 0):.0f} / {o.get('p90', 0):.0f} / {o.get('max', 0):.0f} |
| Prefix-reusable share of prompt tokens | {stats['prefix_reusable_fraction']:.1%} |
| Replay span / last estimated completion | {stats['replay_span_s']:.0f} s / {stats['last_completion_s']:.0f} s |
| Prompts whose token count differs from the source | {stats['prompt_tokens_differ_from_source']} ({stats['questions_padded_to_minimum']} questions padded to {options.min_question_tokens} tokens) |

{"Every request of a role starts with the same " + str(options.document_tokens) + "-token synthesized document (token-exact), then a unique question: the server's prefix cache applies from the second request of each role onward (opt-in ``cache_idle_slots``)." if cached else "Every request carries its own synthesized document: no prefix reuse, the pure prefill-bound case."}
Pacing: {options.pacing.to_json()['rule']} with prefill {options.pacing.prefill_s_per_token} s/token and decode
{options.pacing.decode_s_per_token} s/token (desktop-only worst-case ASSUMPTIONS). Bodies are the realistic builder's
templated text cut to the source token count under each model's chat template. Built by `build_longprompt_trace.py`;
nothing here is a measurement.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mooncake-trace", type=Path, required=True, help="conversation_trace.jsonl")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--trace-name", default=None, help="default mooncake_longprompt_<variant>_v1")
    ap.add_argument("--variant", choices=VARIANTS, default="fresh")
    ap.add_argument("--codec", type=Path, required=True)
    ap.add_argument("--library-dir", type=Path, required=True)
    ap.add_argument("--qwen-tokenizer-model", type=Path, required=True)
    ap.add_argument("--gemma-tokenizer-model", type=Path, required=True)
    ap.add_argument("--llama-tokenizer-model", type=Path, required=True)
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--min-input", type=int, default=2000)
    ap.add_argument("--prompt-cap", type=int, default=8000)
    ap.add_argument("--min-output", type=int, default=4)
    ap.add_argument("--output-cap", type=int, default=512)
    ap.add_argument("--hot-share", type=float, default=0.75)
    ap.add_argument("--think-s", type=float, default=60.0)
    ap.add_argument("--document-tokens", type=int, default=4000)
    ap.add_argument("--min-question-tokens", type=int, default=64)
    ap.add_argument("--prefill-s-per-token", type=float, default=0.02)
    ap.add_argument("--decode-s-per-token", type=float, default=0.65)
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
    options = LongPromptOptions(
        trace_name=args.trace_name or f"mooncake_longprompt_{args.variant}_v1", variant=args.variant, requests=args.requests,
        min_input=args.min_input, prompt_cap=args.prompt_cap, min_output=args.min_output, output_cap=args.output_cap,
        hot_share=args.hot_share, think_s=args.think_s, document_tokens=args.document_tokens,
        min_question_tokens=args.min_question_tokens, seed=args.seed,
        pacing=Pacing(args.prefill_s_per_token, args.decode_s_per_token))
    rows = load_mooncake_rows(args.mooncake_trace)
    qwen_sha, gemma_sha, llama_sha = (digest_file(p) for p in (args.qwen_tokenizer_model, args.gemma_tokenizer_model,
                                                                args.llama_tokenizer_model))
    codecs: dict[str, Codec] = {
        HOT_ROLE: Codec(args.codec, args.qwen_tokenizer_model, qwen_sha, args.library_dir),
        COLD_ROLE: Codec(args.codec, args.gemma_tokenizer_model, gemma_sha, args.library_dir),
        "overlay": Codec(args.codec, args.llama_tokenizer_model, llama_sha, args.library_dir)}
    try:
        large, overlay, stats = build_longprompt_trace(rows, codecs, options, llama_bytes=overlay_identity["artifact_bytes"],
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
    source_sha = digest_file(args.mooncake_trace)
    derivation = {
        "builder": "build_longprompt_trace.py", "source_dataset": SOURCE_DATASET,
        "mooncake_trace": str(args.mooncake_trace), "mooncake_trace_sha256": source_sha, "options": options.to_json(),
        "statistics": stats,
        "tokenizers": {QWEN_TOKENIZER_MODEL: qwen_sha, GEMMA_TOKENIZER_MODEL: gemma_sha, OVERLAY_MODEL_ID: llama_sha},
        "overlay_tokenizer_file": {"path": str(args.llama_tokenizer_model), "sha256": llama_sha,
                                   "artifact_identity_in_rows": overlay_identity},
    }
    write_trace(args.output_dir, options.trace_name, large, overlay, inventory, derivation)
    (args.output_dir / f"{options.trace_name}.md").write_text(markdown(options, stats, args.mooncake_trace, source_sha))
    print(json.dumps(stats, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
