#!/usr/bin/env python3
"""Per-request latency report of one or more runner outputs (read-only).

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.latency_report \
        --run s2a=inputs-two-phone-s2a [--run ...] [--json OUT.json] [--md OUT.md] [--method nearest_rank]

Per request (times on the run's paid clock: 0 = paid_start_ns; arrival = ``replay_arrival_us``):
  ttft_s    first token - arrival                (``first_token_ns`` - paid_start_ns)
  tpot_ms   (end - first token) / (output_tokens - 1)          (client side, includes co-batching)
  e2e_s     end - arrival                         (end = ``completion.actual_end_us``)
  wait_s    execution start - arrival             (start = terminal ``execution_receipt.started_us``; includes
            dispatch queueing AND model loads the request waited for)
  load_s    sum of the terminal ticket's transition receipts (loads this request's route triggered; part of wait)
  prefill_s first token - execution start; with the SSE stream (``streams/request-NNN.raw``, NNN =
            ``combined_request_index``) it splits into server prompt time (``server_prompt_s``) and in-server
            wait for a slot (``server_queue_s``)
  queue_s   everything before the request's own work: wait_s + server_queue_s (wait_s alone without a stream)
  service_s e2e_s - queue_s = server prompt time + decode (end - execution start without a stream)
  server_tpot_ms  ``timings.predicted_per_token_ms`` from the final SSE chunk (cross-check of tpot_ms)

Percentiles: ONE method everywhere, nearest rank (``nearest_rank``): the value at 1-based rank ceil(p/100 x n) of the
sorted sample, p in (0, 100]. It always returns an observed value and never under-reports a tail; with the 14-request
trace p90 is the 13th value and p99 the maximum. ``percentile()`` also implements ``linear`` (numpy default /
R type 7 / statistics.quantiles(method="inclusive")) and ``index_floor`` (sorted[floor(p/100 x n)]) only to audit
numbers quoted earlier (``--audit``); reports use nearest rank.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "ws3-latency-report-v1"
METHODS = ("nearest_rank", "linear", "index_floor")
METRICS = ("ttft_s", "tpot_ms", "e2e_s", "queue_s", "service_s", "wait_s", "load_s", "prefill_s")
PERCENTILES = (50, 90, 99)
TIMINGS = re.compile(r'"timings":(\{[^{}]*\})')


def percentile(values: Iterable[float], p: float, method: str = "nearest_rank") -> float | None:
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    if not 0 < p <= 100:
        raise ValueError("percentile must be in (0, 100]")
    n = len(data)
    if method == "nearest_rank":
        return data[max(1, math.ceil(p / 100 * n - 1e-12)) - 1]
    if method == "linear":
        h = (n - 1) * p / 100
        lo = math.floor(h)
        return data[lo] if lo + 1 >= n else data[lo] + (h - lo) * (data[lo + 1] - data[lo])
    if method == "index_floor":
        return data[min(n - 1, math.floor(p / 100 * n))]
    raise ValueError("unknown percentile method " + method)


def model_role(model_id: str) -> str:
    for role in ("qwen", "gemma", "llama"):
        if role in model_id.lower():
            return role
    return model_id


def find_result(directory: Path) -> Path:
    for candidate in (directory / "run-eval/run/RESULT.json", directory / "RESULT.json", directory / "run/RESULT.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("no RESULT.json under " + str(directory))


def stream_timings(result_path: Path, index: int | None) -> dict[str, Any] | None:
    if index is None:
        return None
    for folder in (result_path.parent / "streams", result_path.parent.parent / "streams"):
        path = folder / ("request-%03d.raw" % index)
        if path.is_file():
            found = TIMINGS.findall(path.read_text(errors="replace"))
            return json.loads(found[-1]) if found else None
    return None


def request_rows(result: Mapping[str, Any], result_path: Path | None = None) -> list[dict[str, Any]]:
    paid_start_ns = result["paid_start_ns"]
    rows = []
    for item in result["request_results"]:
        completion = item.get("completion") or {}
        receipt = completion.get("execution_receipt") or {}
        terminal = item.get("terminal_ticket") or {}
        arrival = item["replay_arrival_us"] / 1e6
        end = completion["actual_end_us"] / 1e6
        first = (item["first_token_ns"] - paid_start_ns) / 1e9 if item.get("first_token_ns") else None
        start = receipt["started_us"] / 1e6 if receipt.get("started_us") is not None else None
        out_tokens = item.get("output_tokens")
        loads = [(t["finished_us"] - t["started_us"]) / 1e6 for t in terminal.get("transition_receipts") or []
                 if t.get("started_us") is not None and t.get("finished_us") is not None]
        row: dict[str, Any] = {
            "request_id": item["request_id"], "model": model_role(item["model_id"]),
            "input_tokens": item.get("input_tokens"), "output_tokens": out_tokens,
            "arrival_s": arrival, "first_token_s": first, "end_s": end, "exec_start_s": start,
            "ttft_s": None if first is None else first - arrival,
            "tpot_ms": None if first is None or not out_tokens or out_tokens < 2 else (end - first) / (out_tokens - 1) * 1e3,
            "e2e_s": end - arrival,
            "wait_s": None if start is None else start - arrival,
            "load_s": sum(loads),
            "service_s": None if start is None else end - start,
            "prefill_s": None if start is None or first is None else first - start,
            "attempts": len(item.get("attempt_ticket_ids") or []),
            "recovered": bool(item.get("recoveries")),
            "executor": (receipt.get("executor_id") or item.get("actual_executor_id") or "").split(":", 1)[-1],
        }
        timings = stream_timings(result_path, item.get("combined_request_index")) if result_path else None
        if timings:
            row["server_prompt_s"] = timings.get("prompt_ms", 0) / 1e3
            row["server_tpot_ms"] = timings.get("predicted_per_token_ms")
            row["server_predicted_n"] = timings.get("predicted_n")
            if row["prefill_s"] is not None:
                row["server_queue_s"] = row["prefill_s"] - row["server_prompt_s"]
        if row["wait_s"] is not None:
            row["queue_s"] = row["wait_s"] + row.get("server_queue_s", 0.0)
            row["service_s"] = row["e2e_s"] - row["queue_s"]
        else:
            row["queue_s"] = None
        rows.append(row)
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], method: str = "nearest_rank") -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for metric in METRICS + ("server_tpot_ms", "server_queue_s"):
        values = [row[metric] for row in rows if row.get(metric) is not None]
        if not values:
            continue
        entry = {"n": len(values), "mean": statistics.fmean(values), "max": max(values)}
        for p in PERCENTILES:
            entry["p%d" % p] = percentile(values, p, method)
        out[metric] = entry
    out["total_wait_s"] = sum(row["wait_s"] for row in rows if row.get("wait_s") is not None)
    out["total_queue_s"] = sum(row["queue_s"] for row in rows if row.get("queue_s") is not None)
    decode = [(row["end_s"] - row["first_token_s"], row["output_tokens"] - 1) for row in rows
              if row.get("first_token_s") is not None and (row.get("output_tokens") or 0) >= 2]
    out["token_weighted_tpot_ms"] = (sum(d for d, _ in decode) / sum(n for _, n in decode) * 1e3) if decode else None
    out["output_tokens"] = sum(row.get("output_tokens") or 0 for row in rows)
    return out


def report_run(label: str, directory: Path, method: str = "nearest_rank") -> dict[str, Any]:
    path = find_result(directory)
    result = json.loads(path.read_text())
    rows = request_rows(result, path)
    models = sorted({row["model"] for row in rows})
    return {"label": label, "status": result.get("status"), "duration_s": result["duration_us"] / 1e6,
            "percentile_method": method, "requests": rows, "overall": summarize(rows, method),
            "by_model": {model: summarize([row for row in rows if row["model"] == model], method) for model in models}}


def audit(rows: Sequence[Mapping[str, Any]], metric: str = "e2e_s") -> dict[str, dict[str, float | None]]:
    values = [row[metric] for row in rows if row.get(metric) is not None]
    return {method: {"p50": percentile(values, 50, method), "p90": percentile(values, 90, method)} for method in METHODS}


def _fmt(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else ("%." + str(digits) + "f") % value


def markdown(reports: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["Percentiles: nearest rank (value at rank ceil(p x n)).", "",
             "| run | scope | n | TTFT p50/p90/p99 s | TPOT mean / p50 / p90 ms | E2E p50/p90/p99 s | E2E mean s |"
             " queue p50/p90 s | service p50/p90 s | total queue s |", "|" + "---|" * 10]
    for label, report in reports.items():
        for scope, summary in [("all", report["overall"])] + sorted(report["by_model"].items()):
            ttft, tpot, e2e, queue, service = (summary.get(k) or {} for k in ("ttft_s", "tpot_ms", "e2e_s", "queue_s", "service_s"))
            lines.append("| %s | %s | %d | %s / %s / %s | %s / %s / %s | %s / %s / %s | %s | %s / %s | %s / %s | %s |" % (
                label, scope, summary["n"], _fmt(ttft.get("p50"), 1), _fmt(ttft.get("p90"), 1), _fmt(ttft.get("p99"), 1),
                _fmt(tpot.get("mean")), _fmt(tpot.get("p50")), _fmt(tpot.get("p90")),
                _fmt(e2e.get("p50")), _fmt(e2e.get("p90")), _fmt(e2e.get("p99")), _fmt(e2e.get("mean")),
                _fmt(queue.get("p50")), _fmt(queue.get("p90")), _fmt(service.get("p50")), _fmt(service.get("p90")),
                _fmt(summary.get("total_queue_s"))))
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="label=run directory")
    parser.add_argument("--method", choices=METHODS, default="nearest_rank")
    parser.add_argument("--audit", action="store_true", help="also print E2E p50/p90 under every method")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args(argv)
    reports = {}
    for spec in args.run:
        label, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit("expected label=path, got " + spec)
        reports[label] = report_run(label, Path(path), args.method)
    text = markdown(reports)
    if args.audit:
        text += "\nE2E percentile audit (seconds):\n"
        for label, report in reports.items():
            text += label + " " + json.dumps({m: {k: None if v is None else round(v, 1) for k, v in d.items()}
                                               for m, d in audit(report["requests"]).items()}) + "\n"
    if args.json:
        args.json.write_text(json.dumps({"schema": SCHEMA, "runs": reports}, indent=1, sort_keys=True) + "\n")
    if args.md:
        args.md.write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
