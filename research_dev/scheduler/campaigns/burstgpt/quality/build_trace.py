#!/usr/bin/env python3
"""Build the GSM8K quality suite: replay-trace shards in the campaign format plus a separate answer key.

    python3 -m research_dev.scheduler.campaigns.burstgpt.quality.build_trace \\
        --gsm8k gsm8k_test.jsonl --split test --output-dir OUT --trace-prefix quality_gsm8k_v1 \\
        --codec llama-token-codec --library-dir LIB \\
        --qwen-tokenizer-model Qwen3-14B-Q4_K_M.gguf --gemma-tokenizer-model gemma-4-12B-it-Q4_0.gguf \\
        --llama-tokenizer-model Llama-3.2-1B-Instruct-Q4_0.gguf \\
        --inventory-from /mnt/storage/burstgpt-source/longtail_eval_v2/TRACE_MANIFEST.json

Every shard directory holds exactly the four files the campaign loader reads (REQUESTS_SEMANTIC_SOURCE.jsonl,
REQUESTS_OVERLAY.jsonl, TRACE_MANIFEST.json, <trace_name>.json) plus a provenance note; a campaign runs one
shard by pointing its `trace` paths at that directory. Each shard carries Llama control rows first, then a
Gemma block, then a Qwen block (one model switch per shard, no interleaving), with dense steady arrivals
paced a little faster than the batched decode service so the servers never idle. Reference answers never
enter a trace: they are in OUT/QUALITY_KEY.jsonl, keyed by request id, next to the rendered prompts
(QUALITY_PROMPTS.jsonl) and the suite manifest (SUITE.json, file sha256s). Items come from one seeded
permutation of the split, so shard k holds the same items in every arm. --pilot builds the one-shard
budget/format pilot from the train split instead (never scored in the confirmatory analysis).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Protocol

from ..build_realistic_trace import (
    GEMMA_TOKENIZER_MODEL, MANIFEST_SCHEMA, OVERLAY_MODEL_ID, OVERLAY_SCHEMA, OVERLAY_STREAM_ID,
    QWEN_TOKENIZER_MODEL, SLO_US, Codec, assign_combined_indices, canonical, digest_bytes, digest_file,
    model_inventory, parse_execution_artifact,
)
from ..trace_inputs import REPLAY_SCHEDULE_SCHEMA, REPLAY_START_US, TRACE_COLD_MODEL_ID, TRACE_HOT_MODEL_ID
from . import protocol
from .gsm8k import (
    EXPECTED_FIRST_TOKENS, EXPECTED_LAST_TOKENS, PROMPT_TEMPLATES, USER_TEMPLATE, Gsm8kItem, QualityDataError,
    load_items, render_prompt,
)

SOURCE_SCHEMA = "research-scheduler-quality-source-v1"
KEY_SCHEMA = "research-scheduler-quality-key-v1"
PROMPTS_SCHEMA = "research-scheduler-quality-prompts-v1"
SUITE_SCHEMA = "research-scheduler-quality-suite-v1"
ROLES = ("llama", "gemma", "qwen")
TOKENIZER_MODELS = {"qwen": QWEN_TOKENIZER_MODEL, "gemma": GEMMA_TOKENIZER_MODEL, "llama": OVERLAY_MODEL_ID}
TRACE_ROLE_IDS = {"qwen": TRACE_HOT_MODEL_ID, "gemma": TRACE_COLD_MODEL_ID}
# measured single-request decode periods (s per token) of the reference run s2a and the batch rows each
# server decodes together (Qwen parallel 4, Gemma parallel 2 in the desktop plan); only used for pacing
DEFAULT_SERVICE_S_PER_TOKEN = {"qwen": 0.50, "gemma": 0.40}
DEFAULT_BATCH_ROWS = {"qwen": 4, "gemma": 2}
ARRIVAL_FACTOR = 0.8
HANDOVER_FACTOR = 0.9
LARGE_START_US = REPLAY_START_US + 4_000_000


class TokenCodec(Protocol):
    def tokenize(self, text: str) -> list[int]: ...

    def validate_detokenize(self, tokens: list[int]) -> None: ...


def assign_items(items: list[Gsm8kItem], *, seed: int, shards: int,
                 per_shard: dict[str, int]) -> list[dict[str, list[Gsm8kItem]]]:
    """Shard s takes the next llama, gemma, then qwen items of one seeded permutation of the split."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    size = sum(per_shard[role] for role in ROLES)
    if shards < 1 or size < 1 or shards * size > len(items):
        raise QualityDataError(f"{shards} shards of {size} items do not fit the {len(items)}-item split")
    result = []
    cursor = 0
    for _ in range(shards):
        shard = {}
        for role in ROLES:
            shard[role] = [items[index] for index in order[cursor:cursor + per_shard[role]]]
            cursor += per_shard[role]
        result.append(shard)
    return result


def arrival_plan(counts: dict[str, int], output_tokens: dict[str, int],
                 service_s_per_token: dict[str, float], batch_rows: dict[str, int]) -> dict[str, list[int]]:
    """Arrival times (us): llama rows 1 s apart from the replay start, then the Gemma block, then the Qwen block.

    Within a block requests arrive every ARRIVAL_FACTOR x the estimated batched service time per request
    (tokens x period / batch rows), so the queue grows slowly and the server never idles; the Qwen block
    starts at HANDOVER_FACTOR x the Gemma block's estimated service end, so Qwen arrives while the last
    Gemma requests decode but never ahead of queued Gemma work in arrival order."""
    arrivals = {"llama": [REPLAY_START_US + 1_000_000 * index for index in range(counts["llama"])]}
    start = max(LARGE_START_US, (arrivals["llama"][-1] + 2_000_000) if arrivals["llama"] else LARGE_START_US)
    for role in ("gemma", "qwen"):
        per_request_s = output_tokens[role] * service_s_per_token[role] / batch_rows[role]
        gap_us = max(1_000_000, int(round(ARRIVAL_FACTOR * per_request_s * 1_000_000)))
        arrivals[role] = [start + gap_us * index for index in range(counts[role])]
        block_service_us = int(round(counts[role] * per_request_s * 1_000_000))
        start = max(start + int(HANDOVER_FACTOR * block_service_us),
                    (arrivals[role][-1] + 1_000_000) if arrivals[role] else start)
    return arrivals


def prompt_tokens(codec: TokenCodec, role: str, text: str,
                  structure: tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]) -> list[int]:
    tokens = codec.tokenize(text)
    first, last = structure[0][role], structure[1][role]
    if tuple(tokens[:len(first)]) != first or tuple(tokens[-len(last):]) != last:
        raise QualityDataError(
            f"{role} prompt tokens do not carry the chat-template structure: starts {tokens[:len(first)]}, "
            f"ends {tokens[-len(last):]} (expected {list(first)} ... {list(last)})")
    codec.validate_detokenize(tokens)
    return tokens


def prompt_sha256(tokens: list[int]) -> str:
    """The runner's request_results prompt_sha256 of the same tokens."""
    return "sha256:" + hashlib.sha256(canonical(tokens)).hexdigest()


def build_shard(*, shard_index: int, trace_name: str, directory: Path, assignment: dict[str, list[Gsm8kItem]],
                codecs: dict[str, TokenCodec], output_tokens: dict[str, int], inventory: dict[str, Any],
                derivation: dict[str, Any], service_s_per_token: dict[str, float], batch_rows: dict[str, int],
                structure=(EXPECTED_FIRST_TOKENS, EXPECTED_LAST_TOKENS)) -> tuple[list[dict], list[dict], dict]:
    """Write one shard's trace files; return its key rows, prompt rows and suite entry."""
    if OVERLAY_MODEL_ID not in inventory:
        raise QualityDataError("model inventory has no overlay (Llama) model")
    if not assignment["llama"] or not assignment["gemma"] or not assignment["qwen"]:
        raise QualityDataError("every shard needs Llama, Gemma and Qwen rows (three-model trace coverage)")
    counts = {role: len(assignment[role]) for role in ROLES}
    arrivals = arrival_plan(counts, output_tokens, service_s_per_token, batch_rows)
    overlay_artifact = inventory[OVERLAY_MODEL_ID]
    large_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    staged = []
    for role in ROLES:
        for item, arrival_us in zip(assignment[role], arrivals[role]):
            text = render_prompt(role, item.question)
            tokens = prompt_tokens(codecs[role], role, text, structure)
            budget = output_tokens[role]
            common = {"arrival_us": arrival_us, "input_tokens": len(tokens), "output_tokens": budget,
                      "prompt_tokenizer_model": TOKENIZER_MODELS[role], "prompt_tokens": tokens, "slo_us": SLO_US,
                      "source_input_tokens": len(tokens), "source_model": "gsm8k-" + item.split,
                      "source_output_tokens": budget, "source_t_us": arrival_us}
            if role == "llama":
                overlay_index = len(overlay_rows)
                row = {**common, "combined_request_index": -1,
                       "event_id": f"{trace_name}:{OVERLAY_STREAM_ID}:{overlay_index:02d}",
                       "execution_model_id": OVERLAY_MODEL_ID, "modality": "text",
                       "model_artifact_bytes": overlay_artifact["artifact_bytes"],
                       "model_artifact_sha256": overlay_artifact["artifact_sha256"], "model_id": OVERLAY_MODEL_ID,
                       "overlay_request_index": overlay_index, "prompt_transport": "tokens",
                       "requested_model_id": OVERLAY_MODEL_ID, "schema": OVERLAY_SCHEMA,
                       "stream_request_index": overlay_index, "trace_stream_id": OVERLAY_STREAM_ID}
                overlay_rows.append(row)
            else:
                request_index = len(large_rows)
                row = {**common, "event_id": f"{trace_name}:{request_index:03d}", "model_id": TRACE_ROLE_IDS[role],
                       "request_index": request_index, "schema": SOURCE_SCHEMA}
                large_rows.append(row)
            staged.append((role, item, text, row))
    order = assign_combined_indices(large_rows, overlay_rows)
    combined_by_event = {}
    arrivals_schedule = []
    for source_kind, source_index, combined in order:
        row = large_rows[source_index] if source_kind == "large" else overlay_rows[source_index]
        if source_kind == "overlay":
            row["combined_request_index"] = combined
        combined_by_event[row["event_id"]] = combined
        arrivals_schedule.append({"combined_request_index": combined, "replay_arrival_us": row["arrival_us"]})

    directory.mkdir(parents=True, exist_ok=False)
    source = b"".join(canonical(row) for row in large_rows)
    overlay = b"".join(canonical(row) for row in overlay_rows)
    (directory / "REQUESTS_SEMANTIC_SOURCE.jsonl").write_bytes(source)
    (directory / "REQUESTS_OVERLAY.jsonl").write_bytes(overlay)
    input_large = sum(row["input_tokens"] for row in large_rows)
    output_large = sum(row["output_tokens"] for row in large_rows)
    input_overlay = sum(row["input_tokens"] for row in overlay_rows)
    output_overlay = sum(row["output_tokens"] for row in overlay_rows)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "base_trace": {"path": str(directory / "REQUESTS_SEMANTIC_SOURCE.jsonl"), "schema": SOURCE_SCHEMA,
                       "sha256": digest_bytes(source), "record_count": len(large_rows),
                       "roles": {"hot": counts["qwen"], "cold": counts["gemma"]},
                       "input_tokens": input_large, "output_tokens": output_large},
        "overlay_trace": {"path": str(directory / "REQUESTS_OVERLAY.jsonl"), "schema": OVERLAY_SCHEMA,
                          "sha256": digest_bytes(overlay), "record_count": len(overlay_rows),
                          "input_tokens": input_overlay, "output_tokens": output_overlay,
                          "arrival_first_us": overlay_rows[0]["arrival_us"],
                          "arrival_last_us": overlay_rows[-1]["arrival_us"], "stream_id": OVERLAY_STREAM_ID},
        "combined_work": {"record_count": len(large_rows) + len(overlay_rows),
                          "input_tokens": input_large + input_overlay, "output_tokens": output_large + output_overlay},
        "execution_models": {"hot": counts["qwen"], "cold": counts["gemma"], OVERLAY_MODEL_ID: len(overlay_rows)},
        "model_inventory": inventory,
        "derivation": {**derivation, "shard_index": shard_index, "trace_name": trace_name,
                       "arrival": {"arrival_factor": ARRIVAL_FACTOR, "handover_factor": HANDOVER_FACTOR,
                                   "service_s_per_token": service_s_per_token, "batch_rows": batch_rows}},
    }
    (directory / "TRACE_MANIFEST.json").write_bytes(canonical(manifest))
    schedule = {"arrivals": arrivals_schedule, "schema": REPLAY_SCHEDULE_SCHEMA, "trace_name": trace_name}
    (directory / f"{trace_name}.json").write_bytes(canonical(schedule))
    span_s = (arrivals_schedule[-1]["replay_arrival_us"] - arrivals_schedule[0]["replay_arrival_us"]) / 1e6
    (directory / f"{trace_name}.md").write_text(
        f"# {trace_name}: GSM8K quality shard {shard_index}\n\n"
        f"Protocol {protocol.PROTOCOL_ID}. {counts['llama']} Llama control rows, then {counts['gemma']} Gemma and "
        f"{counts['qwen']} Qwen rows ({derivation['dataset']['split']} split items of one seeded permutation), "
        f"{output_large + output_overlay} output tokens (fixed budget per request, end of generation banned by the "
        f"runner), {input_large + input_overlay} prompt tokens, arrival span {span_s:.0f} s. Reference answers "
        f"are not in this directory (QUALITY_KEY.jsonl of the suite). Built by quality/build_trace.py; nothing here "
        f"is a measurement.\n")

    key_rows, prompt_rows = [], []
    for role, item, text, row in staged:
        request_id = row["event_id"]
        key_rows.append({"schema": KEY_SCHEMA, "request_id": request_id, "trace_name": trace_name,
                         "shard": shard_index, "role": role, "item_id": item.item_id, "split": item.split,
                         "index": item.index, "reference": format(item.reference.normalize(), "f"),
                         "combined_request_index": combined_by_event[request_id],
                         "prompt_sha256": prompt_sha256(row["prompt_tokens"]),
                         "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
                         "arrival_us": row["arrival_us"]})
        prompt_rows.append({"schema": PROMPTS_SCHEMA, "request_id": request_id, "role": role,
                            "item_id": item.item_id, "prompt_text": text})
    files = {name: digest_file(directory / name) for name in sorted(path.name for path in directory.iterdir())}
    entry = {"index": shard_index, "trace_name": trace_name, "directory": str(directory), "files": files,
             "requests": counts, "output_tokens": sum(row["output_tokens"] for row in (*large_rows, *overlay_rows)),
             "arrival_span_s": span_s}
    return key_rows, prompt_rows, entry


def build_suite(*, items: list[Gsm8kItem], dataset: dict[str, Any], output_dir: Path, trace_prefix: str,
                codecs: dict[str, TokenCodec], inventory: dict[str, Any], seed: int, shards: int,
                per_shard: dict[str, int], output_tokens: dict[str, int], planned_shards: int,
                tokenizers: dict[str, str], service_s_per_token: dict[str, float] | None = None,
                batch_rows: dict[str, int] | None = None, pilot: bool = False,
                structure=(EXPECTED_FIRST_TOKENS, EXPECTED_LAST_TOKENS)) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("SUITE.json", "QUALITY_KEY.jsonl", "QUALITY_PROMPTS.jsonl"):
        if (output_dir / name).exists():
            raise QualityDataError(f"{output_dir / name} exists; choose a fresh output directory")
    service = dict(DEFAULT_SERVICE_S_PER_TOKEN if service_s_per_token is None else service_s_per_token)
    rows = dict(DEFAULT_BATCH_ROWS if batch_rows is None else batch_rows)
    assignment = assign_items(items, seed=seed, shards=shards, per_shard=per_shard)
    templates = {"user": USER_TEMPLATE, **PROMPT_TEMPLATES}
    derivation = {"quality_protocol": protocol.PROTOCOL_ID, "dataset": dataset, "assignment_seed": seed,
                  "per_shard": per_shard, "output_tokens": output_tokens, "tokenizers": tokenizers,
                  "prompt_templates_sha256": digest_bytes(canonical(templates)),
                  "extraction_rule": protocol.EXTRACTION_ID}
    key_rows, prompt_rows, entries = [], [], []
    for shard_index, shard in enumerate(assignment):
        trace_name = f"{trace_prefix}_s{shard_index:02d}"
        key, prompts, entry = build_shard(
            shard_index=shard_index, trace_name=trace_name, directory=output_dir / f"shard-{shard_index:02d}",
            assignment=shard, codecs=codecs, output_tokens=output_tokens, inventory=inventory,
            derivation=derivation, service_s_per_token=service, batch_rows=rows, structure=structure)
        entry["planned"] = shard_index < planned_shards
        key_rows += key
        prompt_rows += prompts
        entries.append(entry)
    key_bytes = b"".join(canonical(row) for row in key_rows)
    prompt_bytes = b"".join(canonical(row) for row in prompt_rows)
    (output_dir / "QUALITY_KEY.jsonl").write_bytes(key_bytes)
    (output_dir / "QUALITY_PROMPTS.jsonl").write_bytes(prompt_bytes)
    suite = {"schema": SUITE_SCHEMA, "protocol_id": protocol.PROTOCOL_ID, "pilot": pilot, "trace_prefix": trace_prefix,
             "dataset": dataset, "assignment_seed": seed, "per_shard": per_shard, "output_tokens": output_tokens,
             "planned_shards": planned_shards, "shards": entries, "templates": templates, "tokenizers": tokenizers,
             "extraction_rule": protocol.EXTRACTION_ID,
             "key_sha256": digest_bytes(key_bytes), "prompts_sha256": digest_bytes(prompt_bytes),
             "requests": len(key_rows)}
    (output_dir / "SUITE.json").write_bytes(canonical(suite))
    return suite


def load_inventory(path: Path) -> dict[str, Any]:
    """model_inventory of an existing trace manifest (the pinned execution artifacts, no re-hashing)."""
    inventory = json.loads(path.read_text(encoding="ascii")).get("model_inventory")
    if (type(inventory) is not dict or len(inventory) != 3 or OVERLAY_MODEL_ID not in inventory
            or not all(type(row) is dict and type(row.get("artifact_bytes")) is int and row["artifact_bytes"] > 0
                       and type(row.get("artifact_sha256")) is str and len(row["artifact_sha256"]) == 64
                       and type(row.get("artifact_file")) is str for row in inventory.values())):
        raise QualityDataError(f"{path} has no three-model inventory with the overlay model")
    return inventory


def _role_values(values: list[str] | None, default: dict[str, Any], kind) -> dict[str, Any]:
    result = dict(default)
    for value in values or ():
        role, _, number = value.partition("=")
        if role not in result or not number:
            raise SystemExit(f"expected ROLE=VALUE with ROLE in {sorted(result)}: {value}")
        result[role] = kind(number)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gsm8k", type=Path, required=True, help="pinned GSM8K jsonl of --split")
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--pilot", action="store_true",
                        help="one-shard budget/format pilot from the train split (PILOT_SEED, PILOT_PER_MODEL)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-prefix", default=None)
    parser.add_argument("--shards", type=int, default=protocol.MAXIMUM_SHARDS)
    parser.add_argument("--output-tokens", action="append", metavar="ROLE=N",
                        help="per-role output budget chosen by `score pilot` (required for qwen, gemma and llama; "
                             f"--pilot always uses {protocol.PILOT_OUTPUT_TOKENS})")
    parser.add_argument("--codec", type=Path, required=True, help="llama-token-codec binary")
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer-model", type=Path, required=True)
    parser.add_argument("--gemma-tokenizer-model", type=Path, required=True)
    parser.add_argument("--llama-tokenizer-model", type=Path, required=True)
    inventory_source = parser.add_mutually_exclusive_group(required=True)
    inventory_source.add_argument("--inventory-from", type=Path,
                                  help="copy model_inventory from this TRACE_MANIFEST.json")
    inventory_source.add_argument("--execution-artifact", type=parse_execution_artifact, action="append",
                                  metavar="MODEL_ID=PATH[=KIND]")
    args = parser.parse_args()
    split = args.split
    if args.pilot and split != "train":
        raise SystemExit("--pilot uses the train split: pass --split train and the train jsonl")
    items = load_items(args.gsm8k, split)
    if args.pilot:
        if args.output_tokens:
            raise SystemExit(f"--pilot always runs {protocol.PILOT_OUTPUT_TOKENS} output tokens")
        output_tokens = {role: protocol.PILOT_OUTPUT_TOKENS for role in ROLES}
        seed, shards, planned = protocol.PILOT_SEED, 1, 1
        per_shard = {role: protocol.PILOT_PER_MODEL for role in ROLES}
    else:
        output_tokens = _role_values(args.output_tokens, {role: None for role in ROLES}, int)
        if any(output_tokens[role] not in protocol.OUTPUT_TOKEN_CANDIDATES for role in ROLES):
            raise SystemExit("--output-tokens ROLE=N is required for qwen, gemma and llama, N one of "
                             f"{protocol.OUTPUT_TOKEN_CANDIDATES} (the `score pilot` choice)")
        seed, shards, planned = protocol.ASSIGNMENT_SEED, args.shards, protocol.PLANNED_SHARDS
        per_shard = {"llama": protocol.LLAMA_PER_SHARD, "gemma": protocol.GEMMA_PER_SHARD,
                     "qwen": protocol.QWEN_PER_SHARD}
    prefix = args.trace_prefix or ("quality_gsm8k_pilot" if args.pilot else "quality_gsm8k_v1")
    inventory = (load_inventory(args.inventory_from) if args.inventory_from is not None
                 else model_inventory(args.execution_artifact))
    dataset = {"name": "gsm8k", "split": split, "sha256": protocol.GSM8K_SHA256[split],
               "url": protocol.GSM8K_URL[split], "commit": protocol.GSM8K_COMMIT, "license": "MIT"}
    models = {"qwen": args.qwen_tokenizer_model, "gemma": args.gemma_tokenizer_model,
              "llama": args.llama_tokenizer_model}
    hashes = {role: digest_file(path) for role, path in models.items()}
    codecs = {role: Codec(args.codec, path, hashes[role], args.library_dir) for role, path in models.items()}
    try:
        suite = build_suite(items=items, dataset=dataset, output_dir=args.output_dir, trace_prefix=prefix,
                            codecs=codecs, inventory=inventory, seed=seed, shards=shards, per_shard=per_shard,
                            output_tokens=output_tokens, planned_shards=planned,
                            tokenizers={TOKENIZER_MODELS[role]: hashes[role] for role in ROLES}, pilot=args.pilot)
    finally:
        for codec in codecs.values():
            codec.close()
    print(json.dumps({"output_dir": str(args.output_dir), "requests": suite["requests"],
                      "shards": len(suite["shards"]), "planned_shards": planned,
                      "key_sha256": suite["key_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
