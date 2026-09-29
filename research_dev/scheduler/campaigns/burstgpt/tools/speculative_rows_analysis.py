#!/usr/bin/env python3
"""Tokens per step and rows per phone call of one runner output directory (speculative rows).

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.speculative_rows_analysis --run-dir run/ [--json]

Reads ``RESULT.json``: every request row that carries ``speculative`` (campaign
``speculative_rows``) gives its draft count, accepted draft, verification steps and tokens per
step; the run summary ``speculative_rows`` is repeated. The desktop server logs
(``large-model-*-desktop.stderr``) give the histogram of rows per phone FFN call from the
``S41SERVERFFNUSB ... tokens=<rows>`` lines: a verification step of one slot with ``k`` draft
tokens shows as ``rows = 1 + k``. A run without the key has no ``speculative`` rows and every
call carries the batched slot count; the histogram is then the batch histogram.
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

SCHEMA = "s42-speculative-rows-analysis-v1"
USB_CALL = re.compile(r"S41SERVERFFNUSB .*?\btokens=(\d+)\b")


def usb_rows_histogram(lines: Iterable[str]) -> dict[int, int]:
    """Rows per phone FFN call -> number of calls, from server stderr lines."""
    counts: Counter[int] = Counter()
    for line in lines:
        match = USB_CALL.search(line)
        if match is not None:
            counts[int(match.group(1))] += 1
    return dict(sorted(counts.items()))


def request_speculative_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-request draft statistics of a RESULT (empty without the campaign key)."""
    rows = []
    for request in result.get("request_results", []):
        speculative = request.get("speculative")
        if type(speculative) is not dict:
            continue
        for name in ("draft_n", "draft_accepted", "verif_steps", "tokens_per_step_ppm"):
            if type(speculative.get(name)) is not int:
                raise ValueError("request speculative statistics are invalid: " + name)
        rows.append({
            "acceptance_rate_ppm": speculative.get("acceptance_rate_ppm", 0),
            "draft_accepted": speculative["draft_accepted"],
            "draft_n": speculative["draft_n"],
            "model_id": request.get("model_id"),
            "n_max": speculative.get("n_max"),
            "output_tokens": request.get("output_tokens"),
            "request_id": request.get("request_id"),
            "tokens_per_step_ppm": speculative["tokens_per_step_ppm"],
            "verif_steps": speculative["verif_steps"],
        })
    return rows


def summarize(run_dir: Path) -> dict[str, Any]:
    """The analysis of one run directory."""
    result = json.loads((run_dir / "RESULT.json").read_text(encoding="utf-8"))
    histograms = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "large-model-*-desktop.stderr"))):
        with open(path, encoding="utf-8", errors="replace") as stream:
            histogram = usb_rows_histogram(stream)
        if histogram:
            histograms[os.path.basename(path)] = histogram
    return {
        "requests": request_speculative_rows(result),
        "rows_per_call": histograms,
        "schema": SCHEMA,
        "summary": result.get("speculative_rows"),
    }


def _print(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    print("== speculative rows summary:", "absent (campaign key not set)" if summary is None else json.dumps(
        {key: value for key, value in summary.items() if key != "configuration"}, sort_keys=True))
    print("== per request: id model out draft_n accepted steps tokens/step")
    for row in report["requests"]:
        print("%s %s %s %d %d %d %.2f" % (
            str(row["request_id"]).split(":")[-1], str(row["model_id"])[:12], row["output_tokens"],
            row["draft_n"], row["draft_accepted"], row["verif_steps"], row["tokens_per_step_ppm"] / 1e6))
    print("== rows per phone call (server: rows=n calls)")
    for name, histogram in report["rows_per_call"].items():
        print(name, " | ".join("rows=%s: %d" % (rows, calls) for rows, calls in histogram.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = summarize(args.run_dir)
    if args.json:
        json.dump(report, sys.stdout, sort_keys=True, indent=1)
        print()
    else:
        _print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
