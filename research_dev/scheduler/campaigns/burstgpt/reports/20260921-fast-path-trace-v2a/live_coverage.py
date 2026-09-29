#!/usr/bin/env python3
"""Report phone token coverage of a trace arm while it is still running.

The launcher only writes RESULT.json at the end, but each server logs one
`S41SERVERFFNCALL context=<hex request id>:... tokens=N` line per phone call, where N is the number
of decode rows in that call, and the release line carries the layer mask. Assisted tokens are the
summed rows divided by the released layer count.

A cohort call carries several requests' rows but names only one of them, so per-request attribution
is not sound and is reported only as the set of requests observed to have driven a call, a lower
bound on assisted requests. The row total is exact because every row appears once.

    python3 live_coverage.py <run dir> [--reference RESULT.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

CALL = re.compile(r"S41SERVERFFNCALL context=([0-9a-f]+):[^ ]* .*?tokens=(\d+)")
MASK = re.compile(r"phase=decode layer_mask=(\d+)")


def released_layers(text: str) -> int:
    masks = {int(value) for value in MASK.findall(text)}
    return max((bin(value).count("1") for value in masks), default=0)


def assisted_rows(text: str) -> collections.Counter:
    rows: collections.Counter = collections.Counter()
    for request_hex, tokens in CALL.findall(text):
        try:
            request_id = bytes.fromhex(request_hex).decode("ascii")
        except ValueError:
            continue
        rows[request_id] += int(tokens)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--reference", type=pathlib.Path,
                        help="RESULT.json of a completed arm of the same trace, for output-token totals")
    args = parser.parse_args()

    rows: collections.Counter = collections.Counter()
    layers = 0
    for log in sorted(args.run.glob("*.stderr")):
        text = log.read_text(errors="replace")
        rows.update(assisted_rows(text))
        layers = max(layers, released_layers(text))
    if not layers:
        print("no release seen yet; cannot convert call rows to tokens")
        return 0

    assisted_tokens = sum(rows.values()) / layers
    print(f"released layers per token: {layers}")
    print(f"assisted decode tokens:   {assisted_tokens:.0f}")
    print(f"requests seen driving a call (lower bound): {len(rows)}")
    print("  " + ", ".join(sorted(request_id.rsplit(":", 1)[-1] for request_id in rows)))

    if args.reference is not None:
        reference = json.loads(args.reference.read_text())["request_results"]
        for prefix in sorted({row["model_id"].split("-")[0] for row in reference}):
            total = sum(row["output_tokens"] for row in reference
                        if row["model_id"].startswith(prefix))
            count = sum(1 for row in reference if row["model_id"].startswith(prefix))
            print(f"trace total for {prefix:<10} {total:>6} output tokens over {count} requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
