#!/usr/bin/env python3
"""Extract per-call phone FFN timings from a trace run.

Host side: every `S41SERVERFFNSHAPE {json}` line in run/large-model-*.stderr (one per (tokens,
columns) shape per server process; compute_mean_ms is the phone worker's own compute clock as
echoed over the DMA-BUF transport). Phone side: `S43DUALFFN request=... layer=... tokens=...
columns=... primary_columns=... secondary_columns=... primary_us=... secondary_us=... wait_us=...
merge_us=... total_us=...` lines and the `[ffn-worker] dual warmup ...` line from the pulled
worker.log files. Prints Markdown tables and writes a JSON summary.

    python3 extract_phone_timings.py --run-dir RUN_DIR [--worker-log PATH ...] --out SUMMARY.json
        [--reference columns=ms ...]
"""
import argparse
import json
import pathlib
import re
import statistics

SHAPE = re.compile(r"S41SERVERFFNSHAPE (\{.*\})")
DUAL = re.compile(r"S43DUALFFN (.*)$")
WARMUP = re.compile(r"\[ffn-worker\] dual warmup (.*)$")
DUAL_P50 = re.compile(r"\[ffn-worker\] dual requests=(\d+) primary_p50_us=(\d+) secondary_p50_us=(\d+) total_p50_us=(\d+)")


def kv(text):
    out = {}
    for token in text.split():
        key, _, value = token.partition("=")
        try:
            out[key] = int(value)
        except ValueError:
            out[key] = value
    return out


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]


def host_shapes(run_dir):
    rows = []
    for path in sorted(run_dir.glob("run/large-model-*.stderr")):
        for line in path.read_text(errors="replace").splitlines():
            match = SHAPE.search(line)
            if match:
                row = json.loads(match.group(1))
                row["process"] = path.name
                rows.append(row)
    return rows


def phone_calls(paths):
    calls, warmups, p50s = [], [], []
    for path in paths:
        for line in pathlib.Path(path).read_text(errors="replace").splitlines():
            match = DUAL.search(line)
            if match:
                row = kv(match.group(1))
                row["log"] = pathlib.Path(path).name
                calls.append(row)
                continue
            match = WARMUP.search(line)
            if match:
                row = kv(match.group(1))
                row["log"] = pathlib.Path(path).name
                warmups.append(row)
                continue
            match = DUAL_P50.search(line)
            if match:
                p50s.append({"log": pathlib.Path(path).name, "requests": int(match.group(1)),
                             "primary_p50_us": int(match.group(2)), "secondary_p50_us": int(match.group(3)),
                             "total_p50_us": int(match.group(4))})
    return calls, warmups, p50s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=pathlib.Path, required=True)
    ap.add_argument("--worker-log", type=pathlib.Path, action="append", default=[])
    ap.add_argument("--reference", action="append", default=[], help="columns=compute_ms of the reference run")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    references = {}
    for item in args.reference:
        columns, _, value = item.partition("=")
        references[int(columns)] = float(value)

    shapes = host_shapes(args.run_dir)
    print("## Host-side S41SERVERFFNSHAPE (per server process, per shape)\n")
    print("| process | tokens | columns | calls | compute_mean_ms | compute_p50_ms | rpc_mean_ms | overlap_mean_ms | reference_ms | speedup |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in shapes:
        reference = references.get(row["columns"])
        speedup = f"{reference / row['compute_mean_ms']:.3f}" if reference else "-"
        print(f"| {row['process']} | {row['tokens']} | {row['columns']} | {row['calls']} | {row['compute_mean_ms']:.3f} | "
              f"{row['compute_p50_ms']:.3f} | {row['rpc_mean_ms']:.3f} | {row['overlap_mean_ms']:.3f} | "
              f"{reference if reference else '-'} | {speedup} |")
    weighted = {}
    for row in shapes:
        bucket = weighted.setdefault((row["tokens"], row["columns"]), {"calls": 0, "compute_sum": 0.0, "rpc_sum": 0.0})
        bucket["calls"] += row["calls"]
        bucket["compute_sum"] += row["calls"] * row["compute_mean_ms"]
        bucket["rpc_sum"] += row["calls"] * row["rpc_mean_ms"]
    print("\n## Host-side call-weighted mean per shape\n")
    print("| tokens | columns | calls | compute_mean_ms | rpc_mean_ms | reference_ms | speedup |")
    print("|---|---|---|---|---|---|---|")
    weighted_rows = []
    for (tokens, columns), bucket in sorted(weighted.items()):
        compute = bucket["compute_sum"] / bucket["calls"]
        rpc = bucket["rpc_sum"] / bucket["calls"]
        reference = references.get(columns)
        weighted_rows.append({"tokens": tokens, "columns": columns, "calls": bucket["calls"], "compute_mean_ms": compute,
                              "rpc_mean_ms": rpc, "reference_ms": reference,
                              "speedup": (reference / compute) if reference else None})
        print(f"| {tokens} | {columns} | {bucket['calls']} | {compute:.3f} | {rpc:.3f} | {reference if reference else '-'} | "
              f"{(reference / compute):.3f} |" if reference else
              f"| {tokens} | {columns} | {bucket['calls']} | {compute:.3f} | {rpc:.3f} | - | - |")

    calls, warmups, p50s = phone_calls(args.worker_log)
    grouped = {}
    for row in calls:
        grouped.setdefault((row["log"], row["tokens"], row["columns"]), []).append(row)
    print("\n## Phone-side S43DUALFFN per worker log and shape (us)\n")
    print("| log | tokens | columns | calls | primary_cols | secondary_cols | total p50 | total mean | total p90 | primary p50 | secondary p50 | wait p50 | merge p50 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    dual_rows = []
    for (log, tokens, columns), rows in sorted(grouped.items()):
        totals = [r["total_us"] for r in rows]
        summary = {
            "log": log, "tokens": tokens, "columns": columns, "calls": len(rows),
            "primary_columns": rows[0]["primary_columns"], "secondary_columns": rows[0]["secondary_columns"],
            "total_p50_us": percentile(totals, 0.5), "total_mean_us": statistics.fmean(totals),
            "total_p90_us": percentile(totals, 0.9),
            "primary_p50_us": percentile([r["primary_us"] for r in rows], 0.5),
            "secondary_p50_us": percentile([r["secondary_us"] for r in rows], 0.5),
            "wait_p50_us": percentile([r["wait_us"] for r in rows], 0.5),
            "merge_p50_us": percentile([r["merge_us"] for r in rows], 0.5),
        }
        dual_rows.append(summary)
        print(f"| {log} | {tokens} | {columns} | {len(rows)} | {summary['primary_columns']} | {summary['secondary_columns']} | "
              f"{summary['total_p50_us']} | {summary['total_mean_us']:.0f} | {summary['total_p90_us']} | {summary['primary_p50_us']} | "
              f"{summary['secondary_p50_us']} | {summary['wait_p50_us']} | {summary['merge_p50_us']} |")
    print("\n## Phone-side warm-up lines\n")
    for row in warmups:
        print("-", json.dumps(row))
    args.out.write_text(json.dumps({
        "host_shapes": shapes, "host_weighted": weighted_rows, "phone_dual": dual_rows,
        "phone_warmups": warmups, "phone_dual_p50_lines": p50s, "references_ms": references,
    }, indent=1, sort_keys=True) + "\n")
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
