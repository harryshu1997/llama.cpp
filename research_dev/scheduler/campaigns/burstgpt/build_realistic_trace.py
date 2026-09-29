#!/usr/bin/env python3
"""Build a realistic replay trace from the BurstGPT conversation logs.

The existing 24-request replay compresses hand-picked requests into a short window. This builder
keeps BurstGPT's own arrival process instead: it takes one contiguous window of the conversation
logs (real inter-arrival times, real request and response token counts), maps ChatGPT rows to the
hot model and GPT-4 rows to the cold model, synthesizes chat prompts of exactly the source token
count with the same templated body the unified trace uses, and writes the four files the campaign
loader consumes:

    REQUESTS_SEMANTIC_SOURCE.jsonl   large-model rows (schema s43-burstgpt-realistic-semantic-source-v1)
    REQUESTS_OVERLAY.jsonl           small-model rows: a share of the window's shortest ChatGPT requests
                                     re-targeted to the Llama overlay model (schema s42-full-fp16-llama1b-overlay-v1)
    TRACE_MANIFEST.json              identities the loader checks (record counts, roles, sha256)
    <trace_name>.json                replay schedule (research-scheduler-burstgpt-replay-v1)
    <trace_name>.md                  provenance and the window's statistics

Row semantics follow the unified trace: the hot role is stored as model_id "gemma-4-12b-it-q8_0"
and carries a Qwen chat prompt, the cold role as "qwen3-14b-q4_k_m" with a Gemma chat prompt; the
campaign's models manifest maps the roles to the executed artifacts. Overlay rows carry the
combined_request_index the loader expects (their position in the arrival-sorted merge of both files).
Token counts above the caps are clipped and the source values kept, exactly like the unified builder.
The pipeline needs at least one overlay row (it takes the small model's id and a warm-up shape from
it), so --small-model-share must leave one. Nothing here is a measurement.

--execution-artifact MODEL_ID=PATH[=KIND] (one per hot, cold and overlay execution model) pins each artifact's
bytes and sha256 into the manifest's model_inventory; the runner refuses to replay against other artifacts.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

from .trace_inputs import REPLAY_SCHEDULE_SCHEMA, REPLAY_START_US, TRACE_COLD_MODEL_ID, TRACE_HOT_MODEL_ID

SOURCE_SCHEMA = "s43-burstgpt-realistic-semantic-source-v1"
MANIFEST_SCHEMA = "s42-full-fp16-llama1b-overlay-manifest-v1"
OVERLAY_SCHEMA = "s42-full-fp16-llama1b-overlay-v1"
OVERLAY_MODEL_ID = "llama-3.2-1b-instruct-q4_0"
OVERLAY_STREAM_ID = "llama-3.2-1b-overlay"
SLO_US = 30_000_000
HOT_SOURCE_MODEL = "ChatGPT"
COLD_SOURCE_MODEL = "GPT-4"
QWEN_TOKENIZER_MODEL = "qwen3-14b-q4_k_m"
GEMMA_TOKENIZER_MODEL = "gemma-4-12b-it-q4_0"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def parse_execution_artifact(spec: str) -> tuple[str, Path, str]:
    """MODEL_ID=PATH[=KIND]; KIND defaults to text_decoder (run7 used text_decoder_f16_proxy for the f16 proxies)."""
    parts = spec.split("=")
    if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("execution artifact must be MODEL_ID=PATH[=KIND]")
    return parts[0], Path(parts[1]), (parts[2] if len(parts) == 3 else "text_decoder")


def model_inventory(specs: list[tuple[str, Path, str]]) -> dict[str, dict[str, Any]]:
    """The manifest's model_inventory: the runner pins every execution model's artifact to it before replay."""
    role_ids = {TRACE_HOT_MODEL_ID, TRACE_COLD_MODEL_ID} & {model_id for model_id, _, _ in specs}
    if role_ids:
        # The runner looks the inventory up by execution model id (e.g. qwen3-14b-q4km-dequant-f16);
        # the trace role ids stored in the rows never match, so the replay would fail after preflight.
        raise SystemExit(f"--execution-artifact names trace role ids {sorted(role_ids)}; use the execution model ids")
    inventory: dict[str, dict[str, Any]] = {}
    for model_id, path, kind in specs:
        if model_id in inventory:
            raise ValueError(f"duplicate execution artifact for {model_id}")
        if not path.is_file():
            raise ValueError(f"execution artifact for {model_id} is not a file: {path}")
        inventory[model_id] = {"artifact_bytes": path.stat().st_size, "artifact_file": path.name,
                               "artifact_sha256": digest_file(path), "kind": kind}
    return inventory


class Codec:
    """The layersplit token codec: one process per tokenizer model, JSON lines over stdio."""

    def __init__(self, executable: Path, model: Path, model_sha256: str, library_dir: Path):
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{library_dir}:" + env.get("LD_LIBRARY_PATH", "")
        self.model_sha256 = model_sha256
        self.request_id = 0
        self.process = subprocess.Popen(
            [str(executable), "--model", str(model), "--model-sha256", model_sha256],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, bufsize=1)

    def exchange(self, op: str, key: str, value: Any) -> dict[str, Any]:
        self.request_id += 1
        request = {key: value, "op": op, "request_id": self.request_id, "schema": "layersplit-token-codec-request-v1"}
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(canonical(request).decode("ascii"))
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("token codec response EOF")
        response = json.loads(line)
        if (response.get("schema") != "layersplit-token-codec-response-v1" or response.get("request_id") != self.request_id
                or response.get("op") != op or response.get("model_sha256") != self.model_sha256):
            raise RuntimeError("token codec response identity")
        return response

    def tokenize(self, text: str) -> list[int]:
        tokens = self.exchange("tokenize", "text", text).get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise RuntimeError("token codec tokenize response")
        return tokens

    def validate_detokenize(self, tokens: list[int]) -> None:
        if not isinstance(self.exchange("detokenize", "tokens", tokens).get("text"), str):
            raise RuntimeError("token codec detokenize response")

    def close(self) -> None:
        assert self.process.stdin is not None and self.process.stderr is not None
        self.process.stdin.close()
        error = self.process.stderr.read()
        status = self.process.wait(timeout=30)
        if status != 0 or error:
            raise RuntimeError(f"token codec failed with status {status}: {error.strip()}")


def prompt_body(index: int, target_tokens: int) -> str:
    lead = (f"Request {index}. Explain in clear English how a desktop computer and a phone can cooperate "
            "on neural-network inference. ")
    context = ("Discuss measured latency, memory bandwidth, computation, overlap, and practical tradeoffs. "
               "Use concrete reasoning and complete sentences. ")
    return lead + context * max(128, (target_tokens + 15) // 16)


def build_prompt(codec: Codec, index: int, target_tokens: int, prefix: str, suffix: str, suffix_bos: int | None) -> list[int]:
    body_tokens = codec.tokenize(prefix + prompt_body(index, target_tokens))
    suffix_tokens = codec.tokenize(suffix)
    if suffix_bos is not None:
        if not suffix_tokens or suffix_tokens[0] != suffix_bos:
            raise RuntimeError("chat suffix BOS mismatch")
        suffix_tokens = suffix_tokens[1:]
    body_count = target_tokens - len(suffix_tokens)
    if body_count <= 0 or len(body_tokens) < body_count:
        raise RuntimeError(f"chat prompt target {target_tokens} is too short or too long")
    result = body_tokens[:body_count] + suffix_tokens
    if len(result) != target_tokens:
        raise RuntimeError("chat prompt token conservation")
    codec.validate_detokenize(result)
    return result


def assign_combined_indices(large: list[dict[str, Any]], overlay: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    """Return (source, source_index, combined_index) in the order trace_inputs.merge_rows produces."""
    merged = [(row["arrival_us"], 0, row["request_index"], "large") for row in large]
    merged += [(row["arrival_us"], 1, row["overlay_request_index"], "overlay") for row in overlay]
    merged.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(source, source_index, combined) for combined, (_, _, source_index, source) in enumerate(merged)]


def load_conversation_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows = []
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Log Type") != "Conversation log" or r.get("Model") not in (HOT_SOURCE_MODEL, COLD_SOURCE_MODEL):
                continue
            try:
                rows.append({"t": float(r["Timestamp"]), "model": r["Model"],
                             "input": int(float(r["Request tokens"])), "output": int(float(r["Response tokens"]))})
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r["t"])
    return rows


def long_tail_stats(rows: list[dict[str, Any]], threshold: int) -> dict[str, Any]:
    """Share of requests, and of output tokens, whose source output exceeds the threshold."""
    outputs = [r["output"] for r in rows]
    tail = [o for o in outputs if o > threshold]
    return {"threshold_tokens": threshold, "requests": len(outputs), "long_requests": len(tail),
            "request_share": len(tail) / len(outputs) if outputs else 0.0,
            "output_token_share": sum(tail) / sum(outputs) if outputs else 0.0}


def same_model_overlaps(rows: list[dict[str, Any]], *, arrival_scale: float, output_cap: int | None,
                        service_s_per_token: float) -> int:
    """Count same-model request pairs that could share a decode batch in the replay.

    A later request overlaps an earlier one of the same source model when it arrives (after the arrival
    scale) before the earlier one would finish decoding alone, estimated as its capped output tokens
    times service_s_per_token. This is a selection heuristic, not a prediction of the scheduler.
    """
    if not rows:
        return 0
    t_first = rows[0]["t"]
    count = 0
    for index, earlier in enumerate(rows):
        tokens = earlier["output"] if output_cap is None else min(earlier["output"], output_cap)
        finishes = (earlier["t"] - t_first) * arrival_scale + tokens * service_s_per_token
        count += sum(1 for later in rows[index + 1:]
                     if later["model"] == earlier["model"] and (later["t"] - t_first) * arrival_scale < finishes)
    return count


def select_window(rows: list[dict[str, Any]], *, duration_s: float, min_requests: int, max_requests: int,
                  start_offset_s: float | None, min_input: int, min_output: int,
                  long_tail_threshold: int | None = None,
                  long_tail_tolerance: float = 0.05,
                  max_output_tokens: int | None = None,
                  output_cap: int | None = None,
                  min_same_model_overlaps: int | None = None,
                  min_requests_per_model: int = 0,
                  arrival_scale: float = 1.0,
                  service_s_per_token: float = 0.5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the first window (from the offset, or scanning) whose eligible request count is in range.

    With long_tail_threshold, the window must also be representative of the whole log's long tail:
    its share of requests and its share of output tokens above the threshold must each lie within
    long_tail_tolerance of the log-wide shares. The first such window in scan order is taken, so the
    rule cannot pick the window that happens to favour the treatment.

    With max_output_tokens, the window's output tokens after output_cap must not exceed it: this
    bounds the replay's run time for development traces without changing the selection order.

    With min_same_model_overlaps, the window must contain at least that many same-model request pairs
    that overlap in the replay (see same_model_overlaps), and min_requests_per_model requests of each
    source model, so a development trace exercises batching and model switching.
    """
    usable = [r for r in rows if r["input"] >= min_input and r["output"] >= min_output]
    times = [r["t"] for r in usable]
    t0 = times[0]
    starts = [t0 + start_offset_s] if start_offset_s is not None else [t0 + k * duration_s for k in range(int((times[-1] - t0) // duration_s))]
    log_tail = long_tail_stats(usable, long_tail_threshold) if long_tail_threshold is not None else None
    for start in starts:
        lo = bisect.bisect_left(times, start)
        hi = bisect.bisect_left(times, start + duration_s)
        if not min_requests <= hi - lo <= max_requests:
            continue
        if max_output_tokens is not None:
            capped = sum(r["output"] if output_cap is None else min(r["output"], output_cap)
                         for r in usable[lo:hi])
            if capped > max_output_tokens:
                continue
        if min_requests_per_model and any(
                sum(1 for r in usable[lo:hi] if r["model"] == model) < min_requests_per_model
                for model in (HOT_SOURCE_MODEL, COLD_SOURCE_MODEL)):
            continue
        overlaps = None
        if min_same_model_overlaps is not None:
            overlaps = same_model_overlaps(usable[lo:hi], arrival_scale=arrival_scale, output_cap=output_cap,
                                           service_s_per_token=service_s_per_token)
            if overlaps < min_same_model_overlaps:
                continue
        info = {"window_start_source_s": start, "window_duration_s": duration_s,
                "scanned_windows": len(starts), "eligible_rows": len(usable)}
        if max_output_tokens is not None:
            info["max_output_tokens"] = max_output_tokens
        if overlaps is not None:
            info["concurrency"] = {"same_model_overlaps": overlaps, "minimum": min_same_model_overlaps,
                                   "min_requests_per_model": min_requests_per_model,
                                   "arrival_scale": arrival_scale, "service_s_per_token": service_s_per_token}
        if log_tail is not None:
            tail = long_tail_stats(usable[lo:hi], long_tail_threshold)
            if (abs(tail["request_share"] - log_tail["request_share"]) > long_tail_tolerance
                    or abs(tail["output_token_share"] - log_tail["output_token_share"]) > long_tail_tolerance):
                continue
            info["long_tail"] = {"log": log_tail, "window": tail, "tolerance": long_tail_tolerance,
                                 "rule": "first window whose request and output-token long-tail shares "
                                         "are each within tolerance of the log-wide shares"}
        return usable[lo:hi], info
    raise SystemExit(f"no {duration_s:.0f} s window with {min_requests}..{max_requests} eligible requests"
                     + (" and a representative long tail" if log_tail is not None else "")
                     + (f" and <= {max_output_tokens} capped output tokens" if max_output_tokens is not None else "")
                     + (f" and >= {min_same_model_overlaps} same-model overlaps" if min_same_model_overlaps is not None else "")
                     + "; adjust --min-requests/--max-requests, --long-tail-tolerance or --start-offset")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--burstgpt-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--trace-name", default="burstgpt_realistic30_v1")
    ap.add_argument("--codec", type=Path, required=True, help="llama-token-codec binary")
    ap.add_argument("--library-dir", type=Path, required=True)
    ap.add_argument("--qwen-tokenizer-model", type=Path, required=True, help="Qwen3-14B-Q4_K_M.gguf")
    ap.add_argument("--gemma-tokenizer-model", type=Path, required=True, help="gemma-4-12B-it-Q4_0 gguf")
    ap.add_argument("--llama-tokenizer-model", type=Path, required=True, help="Llama-3.2-1B-Instruct-Q4_0.gguf (overlay model)")
    ap.add_argument("--small-model-share", type=float, default=0.15,
                    help="fraction of the window's ChatGPT rows, shortest outputs first, executed on the small model")
    ap.add_argument("--duration-s", type=float, default=1800.0)
    ap.add_argument("--min-requests", type=int, default=20)
    ap.add_argument("--max-requests", type=int, default=32)
    ap.add_argument("--start-offset", type=float, default=None, help="seconds after the first conversation row; default scans")
    ap.add_argument("--arrival-scale", type=float, default=1.0, help="multiply inter-arrival times (1.0 = BurstGPT's own)")
    ap.add_argument("--prompt-cap", type=int, default=2048)
    ap.add_argument("--output-cap", type=int, default=512)
    ap.add_argument("--min-input", type=int, default=16)
    ap.add_argument("--min-output", type=int, default=4)
    ap.add_argument("--long-tail-threshold", type=int, default=None,
                    help="require the window's share of requests (and output tokens) above this many source "
                         "output tokens to match the whole log's within --long-tail-tolerance")
    ap.add_argument("--long-tail-tolerance", type=float, default=0.05)
    ap.add_argument("--min-same-model-overlaps", type=int, default=None,
                    help="require this many same-model request pairs that overlap in the replay (after "
                         "--arrival-scale), so a development trace exercises batching")
    ap.add_argument("--min-requests-per-model", type=int, default=0,
                    help="require this many requests of each source model in the window")
    ap.add_argument("--overlap-service-s-per-token", type=float, default=0.5,
                    help="single-stream decode seconds per token used by the overlap estimate")
    ap.add_argument("--max-output-tokens", type=int, default=None,
                    help="skip windows whose output tokens (after --output-cap) exceed this; bounds a "
                         "development trace's run time")
    ap.add_argument("--execution-artifact", type=parse_execution_artifact, action="append", required=True,
                    metavar="MODEL_ID=PATH[=KIND]",
                    help="one per execution model (hot, cold, overlay); the manifest pins their bytes and sha256")
    args = ap.parse_args()
    inventory = model_inventory(args.execution_artifact)
    if OVERLAY_MODEL_ID not in inventory or len(inventory) < 3:
        raise SystemExit("--execution-artifact must name the hot, cold and overlay execution models")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name in ("REQUESTS_SEMANTIC_SOURCE.jsonl", "REQUESTS_OVERLAY.jsonl", "TRACE_MANIFEST.json", f"{args.trace_name}.json"):
        if (out / name).exists():
            raise SystemExit(f"{out / name} already exists; choose a fresh output directory")

    rows = load_conversation_rows(args.burstgpt_csv)
    window, window_info = select_window(rows, duration_s=args.duration_s, min_requests=args.min_requests,
                                        max_requests=args.max_requests, start_offset_s=args.start_offset,
                                        min_input=args.min_input, min_output=args.min_output,
                                        long_tail_threshold=args.long_tail_threshold,
                                        long_tail_tolerance=args.long_tail_tolerance,
                                        max_output_tokens=args.max_output_tokens,
                                        output_cap=args.output_cap,
                                        min_same_model_overlaps=args.min_same_model_overlaps,
                                        min_requests_per_model=args.min_requests_per_model,
                                        arrival_scale=args.arrival_scale,
                                        service_s_per_token=args.overlap_service_s_per_token)

    qwen_sha = digest_file(args.qwen_tokenizer_model)
    gemma_sha = digest_file(args.gemma_tokenizer_model)
    llama_sha = digest_file(args.llama_tokenizer_model)
    llama_bytes = args.llama_tokenizer_model.stat().st_size
    hot_codec = Codec(args.codec, args.qwen_tokenizer_model, qwen_sha, args.library_dir)
    cold_codec = Codec(args.codec, args.gemma_tokenizer_model, gemma_sha, args.library_dir)
    llama_codec = Codec(args.codec, args.llama_tokenizer_model, llama_sha, args.library_dir)

    # Small-model share: the ChatGPT rows with the shortest outputs, so the overlay stays the
    # short-request stream it was in the unified trace.
    hot_positions = [i for i, r in enumerate(window) if r["model"] == HOT_SOURCE_MODEL]
    n_small = int(round(len(hot_positions) * args.small_model_share))
    if n_small < 1:
        raise SystemExit("--small-model-share must leave at least one overlay row; the pipeline needs one")
    small_positions = set(sorted(hot_positions, key=lambda i: (window[i]["output"], window[i]["input"], i))[:n_small])

    large_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    clipped = {"input": 0, "output": 0}
    t_first = window[0]["t"]
    try:
        for index, r in enumerate(window):
            input_tokens = min(r["input"], args.prompt_cap)
            output_tokens = min(r["output"], args.output_cap)
            clipped["input"] += r["input"] > args.prompt_cap
            clipped["output"] += r["output"] > args.output_cap
            replay_us = REPLAY_START_US + int(round((r["t"] - t_first) * args.arrival_scale * 1_000_000))
            if index in small_positions:
                overlay_index = len(overlay_rows)
                prompt = build_prompt(llama_codec, index, input_tokens,
                                      "<|start_header_id|>user<|end_header_id|>\n\n",
                                      "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 128000)
                overlay_rows.append({
                    "arrival_us": replay_us,
                    "combined_request_index": -1,  # assigned after the merge order is known
                    "event_id": f"{args.trace_name}:{OVERLAY_STREAM_ID}:{overlay_index:02d}",
                    "execution_model_id": OVERLAY_MODEL_ID,
                    "input_tokens": input_tokens,
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
                    "source_input_tokens": r["input"],
                    "source_model": r["model"],
                    "source_output_tokens": r["output"],
                    "source_t_us": int(round(r["t"] * 1_000_000)),
                    "stream_request_index": overlay_index,
                    "trace_stream_id": OVERLAY_STREAM_ID,
                })
                continue
            if r["model"] == HOT_SOURCE_MODEL:
                role_model_id = TRACE_HOT_MODEL_ID
                prompt = build_prompt(hot_codec, index, input_tokens, "<|im_start|>user\n",
                                      "\n<|im_end|>\n<|im_start|>assistant\n", None)
                tokenizer_model = QWEN_TOKENIZER_MODEL
            else:
                role_model_id = TRACE_COLD_MODEL_ID
                prompt = build_prompt(cold_codec, index, input_tokens, "<|turn>user\n",
                                      "\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", 2)
                tokenizer_model = GEMMA_TOKENIZER_MODEL
            large_rows.append({
                "arrival_us": replay_us,
                "event_id": f"{args.trace_name}:{len(large_rows):03d}",
                "input_tokens": input_tokens,
                "model_id": role_model_id,
                "output_tokens": output_tokens,
                "prompt_tokenizer_model": tokenizer_model,
                "prompt_tokens": prompt,
                "request_index": len(large_rows),
                "schema": SOURCE_SCHEMA,
                "slo_us": SLO_US,
                "source_input_tokens": r["input"],
                "source_model": r["model"],
                "source_output_tokens": r["output"],
                "source_t_us": int(round(r["t"] * 1_000_000)),
            })
    finally:
        hot_codec.close()
        cold_codec.close()
        llama_codec.close()

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
            "roles": {"hot": hot_count, "cold": cold_count},
            "input_tokens": input_total, "output_tokens": output_total,
        },
        "overlay_trace": {
            "path": str(out / "REQUESTS_OVERLAY.jsonl"), "schema": OVERLAY_SCHEMA,
            "sha256": digest_bytes(overlay), "record_count": len(overlay_rows),
            "input_tokens": overlay_input, "output_tokens": overlay_output,
            "arrival_first_us": overlay_rows[0]["arrival_us"], "arrival_last_us": overlay_rows[-1]["arrival_us"],
            "stream_id": OVERLAY_STREAM_ID,
        },
        "combined_work": {"record_count": len(window), "input_tokens": input_total + overlay_input,
                          "output_tokens": output_total + overlay_output},
        "execution_models": {"hot": hot_count, "cold": cold_count, OVERLAY_MODEL_ID: len(overlay_rows)},
        "model_inventory": inventory,
        "derivation": {
            "burstgpt_csv": str(args.burstgpt_csv), "burstgpt_csv_sha256": digest_file(args.burstgpt_csv),
            "log_type": "Conversation log", "model_map": {HOT_SOURCE_MODEL: "hot", COLD_SOURCE_MODEL: "cold"},
            "window": window_info, "arrival_scale": args.arrival_scale,
            "prompt_cap": args.prompt_cap, "output_cap": args.output_cap, "clipped": clipped,
            "tokenizers": {QWEN_TOKENIZER_MODEL: qwen_sha, GEMMA_TOKENIZER_MODEL: gemma_sha, OVERLAY_MODEL_ID: llama_sha},
            "small_model_share": args.small_model_share,
        },
    }
    (out / "TRACE_MANIFEST.json").write_bytes(canonical(manifest))
    schedule = {"arrivals": arrivals, "schema": REPLAY_SCHEDULE_SCHEMA, "trace_name": args.trace_name}
    (out / f"{args.trace_name}.json").write_bytes(canonical(schedule))

    inputs = [row["input_tokens"] for row in (*large_rows, *overlay_rows)]
    outputs = [row["output_tokens"] for row in (*large_rows, *overlay_rows)]
    gaps = [(b["t"] - a["t"]) * args.arrival_scale for a, b in zip(window, window[1:])]
    span_s = (arrivals[-1]["replay_arrival_us"] - arrivals[0]["replay_arrival_us"]) / 1e6
    tail = window_info.get("long_tail")
    long_tail_md = "" if tail is None else (
        f"Long-tail rule: {tail['rule']} (tolerance {tail['tolerance']:.2f}). Requests with source output "
        f"> {tail['log']['threshold_tokens']} tokens: log {tail['log']['request_share']:.1%} of requests / "
        f"{tail['log']['output_token_share']:.1%} of output tokens; this window "
        f"{tail['window']['request_share']:.1%} / {tail['window']['output_token_share']:.1%} "
        f"({tail['window']['long_requests']} of {tail['window']['requests']}).\n")
    concurrency = window_info.get("concurrency")
    if concurrency is not None:
        long_tail_md += (f"Concurrency rule: >= {concurrency['minimum']} same-model request pairs overlapping in the "
                         f"replay (arrival scale {concurrency['arrival_scale']}, {concurrency['service_s_per_token']} s "
                         f"per token single-stream) and >= {concurrency['min_requests_per_model']} requests per model; "
                         f"this window has {concurrency['same_model_overlaps']}.\n")
    if args.max_output_tokens is not None:
        long_tail_md += (f"Development bound: windows with more than {args.max_output_tokens} output tokens "
                         f"(after the output cap) were skipped; this window has {output_total + overlay_output}.\n")
    md = f"""# {args.trace_name}: a BurstGPT conversation-log window replayed at its own pace

Source: BurstGPT conversation logs (`{args.burstgpt_csv.name}`, sha256 `{manifest['derivation']['burstgpt_csv_sha256'][:16]}...`),
one contiguous window of {args.duration_s:.0f} s starting {window_info['window_start_source_s'] - rows[0]['t']:.0f} s after the
first conversation row (scanned {window_info['scanned_windows']} windows for {args.min_requests} to {args.max_requests} eligible
requests). Arrival scale {args.arrival_scale}; caps prompt {args.prompt_cap} / output {args.output_cap} tokens
(clipped {clipped['input']} prompts, {clipped['output']} outputs; source values kept in the rows).
{long_tail_md}
| | Value |
| --- | ---: |
| Requests | {len(window)}: {hot_count} ChatGPT -> hot (Qwen), {cold_count} GPT-4 -> cold (Gemma), {len(overlay_rows)} shortest ChatGPT -> small model (Llama 1B overlay) |
| Replay span | {span_s:.0f} s |
| Inter-arrival p50 / p90 / max | {statistics.median(gaps) if gaps else 0:.1f} / {sorted(gaps)[int(len(gaps) * 0.9)] if gaps else 0:.1f} / {max(gaps) if gaps else 0:.1f} s |
| Prompt tokens p50 / p90 / max | {statistics.median(inputs):.0f} / {sorted(inputs)[int(len(inputs) * 0.9)]} / {max(inputs)} |
| Output tokens p50 / p90 / max | {statistics.median(outputs):.0f} / {sorted(outputs)[int(len(outputs) * 0.9)]} / {max(outputs)} |
| Total prompt / output tokens | {input_total + overlay_input} / {output_total + overlay_output} (overlay {overlay_input} / {overlay_output}) |

Prompts are the unified trace's templated body cut to the source token count with each model's chat
template, tokenized by the codec against the pinned tokenizer models. The overlay rows are the window's
shortest ChatGPT requests executed on the small model, with the Llama 3 chat template.
The replay preserves BurstGPT's inter-arrival times (times the arrival scale); no route, fraction or
transition is prescribed. Built by `build_realistic_trace.py`; nothing here is a measurement.
"""
    (out / f"{args.trace_name}.md").write_text(md)
    print(json.dumps({"requests": len(window), "hot": hot_count, "cold": cold_count, "overlay": len(overlay_rows),
                      "input_tokens": input_total + overlay_input, "output_tokens": output_total + overlay_output,
                      "replay_span_s": span_s, "clipped": clipped,
                      "source_sha256": manifest["base_trace"]["sha256"], "output_dir": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
