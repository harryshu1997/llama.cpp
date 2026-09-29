#!/usr/bin/env python3
"""Run the pinned 64-item MMLU task screen on Gemma CPU or CPU plus OP15."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import threading
import time
from typing import Any
import urllib.request

import run_server_trace
import run_trace


CONFIRMATION = "RUN_GEMMA_OP15_MMLU64_PROBE"
CORPUS_SHA256 = (
    "3ffafee1615ae2de690a2726b880823e"
    "167a3d9c210c5faed86d8f0e93ecff4f"
)


def prompt_for(item: dict[str, Any]) -> str:
    return (
        f"Question: {item['question']}\n"
        f"A. {item['choices'][0]}\n"
        f"B. {item['choices'][1]}\n"
        f"C. {item['choices'][2]}\n"
        f"D. {item['choices'][3]}\n"
        "Answer with exactly one uppercase letter: A, B, C, or D.\n"
        "Answer:"
    )


def parse_answer(content: str) -> str | None:
    match = re.fullmatch(r"\s*([A-D])\s*", content)
    return match.group(1) if match is not None else None


def terminal_ffn_summary(
    lines: list[str], required: bool
) -> dict[str, Any] | None:
    matches = [line for line in lines if line.startswith("S41SERVERFFN {")]
    run_trace.require(
        len(matches) == (1 if required else 0),
        "server FFN summary count",
    )
    return json.loads(matches[0].split(" ", 1)[1]) if matches else None


def completion(port: int, item: dict[str, Any], output: Path) -> dict[str, Any]:
    body = {
        "cache_prompt": False,
        "grammar": "root ::= [A-D]",
        "n_predict": 1,
        "prompt": prompt_for(item),
        "seed": item["item_index"],
        "stream": False,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=run_trace.canonical(body),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ns = time.monotonic_ns()
    with urllib.request.urlopen(request, timeout=3600) as response:
        raw = response.read()
    completed_ns = time.monotonic_ns()
    output.write_bytes(raw)
    value = json.loads(raw)
    run_trace.require(type(value) is dict and "error" not in value, "completion")
    content = value.get("content")
    timings = value.get("timings")
    run_trace.require(
        type(content) is str and type(timings) is dict and
        timings.get("predicted_n") == 1,
        "completion accounting",
    )
    return {
        "answer": parse_answer(content),
        "completed_ns": completed_ns,
        "content": content,
        "decode_ms": timings.get("predicted_ms"),
        "prefill_ms": timings.get("prompt_ms"),
        "started_ns": started_ns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "op15"), required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cold-server", type=Path, required=True)
    parser.add_argument("--cold-model", type=Path, required=True)
    parser.add_argument("--cold-lib-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--bridge-port", type=int, default=25660)
    parser.add_argument("--cold-port", type=int, default=18481)
    parser.add_argument("--cold-ctx-size", type=int, default=32768)
    parser.add_argument("--cold-batch-size", type=int, default=2048)
    parser.add_argument("--cold-ubatch-size", type=int, default=512)
    parser.add_argument("--cold-threads", type=int, default=-1)
    parser.add_argument("--cold-repack", choices=("default", "off"), default="default")
    parser.add_argument("--split-io", choices=("f16", "f32"), default="f16")
    parser.add_argument("--max-columns", type=int, default=11136)
    parser.add_argument(
        "--policy-id",
        choices=tuple(run_server_trace.SPLIT_POLICIES),
        default="i1-balanced",
    )
    parser.add_argument(
        "--split-policy", default=run_server_trace.BALANCED_SPLIT_POLICY
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    run_trace.require(args.execute and args.confirm == CONFIRMATION, "confirmation")
    run_trace.require(args.output.is_absolute() and not args.output.exists(), "output")
    run_server_trace.validate_split_policy(
        args.policy_id, args.max_columns, args.split_policy
    )
    for path in (args.corpus, args.cold_server, args.cold_model, args.cold_lib_dir):
        run_trace.require(path.exists(), f"missing path: {path}")
    if args.mode == "op15":
        run_trace.require(args.bridge is not None and args.bridge.exists(), "bridge")
    run_trace.require(
        run_trace.digest_file(args.corpus) == CORPUS_SHA256,
        "corpus identity",
    )
    run_trace.require(
        args.cold_model.stat().st_size == 6_975_878_176 and
        run_trace.digest_file(args.cold_model) == run_trace.COLD_MODEL_SHA256,
        "cold model identity",
    )
    corpus = run_trace.read_jsonl(args.corpus)
    run_trace.require(
        len(corpus) == 64 and all(
            row["item_index"] == index and
            row["expected_answer"] in "ABCD" and
            type(row["choices"]) is list and len(row["choices"]) == 4
            for index, row in enumerate(corpus)
        ),
        "corpus geometry",
    )

    args.cold_parallel = 8
    args.control_cpus = None
    args.hot_cpus = None
    args.cold_cpus = None
    args.bridge_cpus = None
    args.output.mkdir(parents=True)
    bridge = None
    cold = None
    result = None
    failure = None
    try:
        if args.mode == "op15":
            bridge = run_trace.start_bridge(args, args.output)
        cold = run_server_trace.start_cold_server(args, args.output)
        completion(args.cold_port, corpus[0], args.output / "warm.raw")
        paid_start_ns = time.monotonic_ns()
        rows = []
        for group in range(8):
            items = corpus[group * 8:(group + 1) * 8]
            barrier = threading.Barrier(8)

            def run_item(item: dict[str, Any]) -> dict[str, Any]:
                barrier.wait(timeout=30)
                value = completion(
                    args.cold_port,
                    item,
                    args.output / f"item-{item['item_index']:02d}.raw",
                )
                return {
                    **value,
                    "correct": value["answer"] == item["expected_answer"],
                    "expected_answer": item["expected_answer"],
                    "item_index": item["item_index"],
                    "subject": item["subject"],
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(run_item, item) for item in items]
                rows.extend(future.result(timeout=3600) for future in futures)
        paid_end_ns = time.monotonic_ns()
        runtime_manifest = run_trace.process_runtime_manifest(
            cold.pid, [args.cold_lib_dir, args.cold_server.parent]
        )
        cold.terminate()
        ffn_summary = terminal_ffn_summary(
            cold.stderr_lines, args.mode == "op15"
        )
        shape_summaries = [
            json.loads(line[len("S41SERVERFFNSHAPE "):])
            for line in cold.stderr_lines
            if line.startswith("S41SERVERFFNSHAPE ")
        ]
        if bridge is not None:
            bridge.terminate()
        bridge_summary = run_server_trace.prefixed_json(
            bridge.stderr_lines if bridge is not None else [],
            "FFNDMABUF ",
            args.mode == "op15",
        )
        if args.mode == "op15":
            run_trace.require(
                ffn_summary is not None and bridge_summary is not None and
                ffn_summary["status"] == "ok" and
                bridge_summary["status"] == "ok" and
                ffn_summary["calls"] == bridge_summary["calls"] and
                bridge_summary["reset_recoveries"] == 0,
                "phone summary",
            )
        rows.sort(key=lambda row: row["item_index"])
        parseable = sum(row["answer"] is not None for row in rows)
        correct = sum(row["correct"] for row in rows)
        result = {
            "correct": correct,
            "corpus_sha256": CORPUS_SHA256,
            "duration_s": (paid_end_ns - paid_start_ns) / 1e9,
            "items": rows,
            "mode": args.mode,
            "parseable": parseable,
            "phone": {
                "bridge": bridge_summary,
                "ffn": ffn_summary,
                "shapes": shape_summaries,
            },
            "schema": "s41-gemma-op15-mmlu64-v1",
            "server_runtime_manifest": runtime_manifest,
            "split_policy": {
                "id": args.policy_id,
                "table": args.split_policy,
            },
            "status": "PASS",
        }
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        if cold is not None:
            cold.terminate()
        if bridge is not None:
            bridge.terminate()

    if failure is not None:
        run_trace.write_json(args.output / "FAILURE.json", {
            "error": failure,
            "mode": args.mode,
            "schema": "s41-gemma-op15-mmlu64-failure-v1",
            "status": "FAIL",
        })
        return 2
    run_trace.require(result is not None, "missing result")
    run_trace.write_json(args.output / "RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
