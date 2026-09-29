#!/usr/bin/env python3
"""Summarize ROUTEJSON records from full-model LayerSplit runs."""

import argparse
import json
import statistics
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_records(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "ROUTEJSON "
        where = line.find(marker)
        if where >= 0:
            records.append(json.loads(line[where + len(marker):]))
    if not records:
        raise ValueError(f"no ROUTEJSON records in {path}")
    return records


def common_prefix_length(left, right):
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def summarize(label, path):
    records = load_records(path)
    request_ms = [row["request_wall_us"] / 1000.0 for row in records]
    prefill_ms = [row["prefill_us"] / 1000.0 for row in records]
    decode_ms = [row["decode_us"] / 1000.0 for row in records]
    decode_steps = [max(row["generated_tokens"] - 1, 0) for row in records]
    total_decode_s = sum(row["decode_us"] for row in records) / 1e6
    result = {
        "label": label,
        "path": str(path),
        "route": records[0]["route"],
        "requests": len(records),
        "prompt_tokens": sorted({row["prompt_tokens"] for row in records}),
        "generated_tokens": sum(row["generated_tokens"] for row in records),
        "decode_steps": sum(decode_steps),
        "decode_tokens_per_second": sum(decode_steps) / total_decode_s,
        "request_ms_p50": statistics.median(request_ms),
        "request_ms_p90": percentile(request_ms, 0.90),
        "prefill_ms_p50": statistics.median(prefill_ms),
        "decode_ms_p50": statistics.median(decode_ms),
        "stage_a_ms_p50": statistics.median(
            row["stage_a_us"] / 1000.0 for row in records
        ),
        "host_ms_p50": statistics.median(
            row["host_us"] / 1000.0 for row in records
        ),
        "token_ids": [row["token_ids"] for row in records],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", action="append", required=True,
                        help="LABEL=LOG_PATH; repeat for each route")
    parser.add_argument("--output")
    args = parser.parse_args()

    summaries = []
    for item in args.route:
        label, separator, path = item.partition("=")
        if not separator or not label or not path:
            parser.error("--route must be LABEL=LOG_PATH")
        summaries.append(summarize(label, path))

    baseline = summaries[0]
    baseline_tokens = baseline["token_ids"]
    for summary in summaries:
        comparable = min(len(baseline_tokens), len(summary["token_ids"]))
        matches = sum(
            baseline_tokens[index] == summary["token_ids"][index]
            for index in range(comparable)
        )
        common_prefixes = [
            common_prefix_length(baseline_tokens[index], summary["token_ids"][index])
            for index in range(comparable)
        ]
        token_positions_compared = sum(
            min(len(baseline_tokens[index]), len(summary["token_ids"][index]))
            for index in range(comparable)
        )
        token_positions_matching = sum(
            left_token == right_token
            for index in range(comparable)
            for left_token, right_token in zip(
                baseline_tokens[index], summary["token_ids"][index]
            )
        )
        summary["request_token_sequence_matches"] = matches
        summary["request_token_sequence_compared"] = comparable
        summary["common_prefix_tokens_min"] = min(common_prefixes)
        summary["common_prefix_tokens_p50"] = statistics.median(common_prefixes)
        summary["token_positions_matching"] = token_positions_matching
        summary["token_positions_compared"] = token_positions_compared
        summary["request_speedup_vs_first"] = (
            baseline["request_ms_p50"] / summary["request_ms_p50"]
        )

    output = {
        "schema": "s41-full-model-cpu-op15-analysis-v1",
        "baseline": baseline["label"],
        "routes": summaries,
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
