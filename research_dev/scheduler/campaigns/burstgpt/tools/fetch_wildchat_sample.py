#!/usr/bin/env python3
"""Fetch a clean multi-turn conversation sample from allenai/WildChat-1M (WS24 conversation trace source).

Uses the Hugging Face datasets-server rows API (100 rows per call, no parquet tooling needed):
    https://datasets-server.huggingface.co/rows?dataset=allenai/WildChat-1M&config=default&split=train&offset=N&length=100
Rows are kept when language == English (or --language), turn >= --min-turns, not toxic, not redacted, and
(optionally) model matches --model. The sample is written as a JSON list of the raw rows plus a provenance
file (dataset, offsets, fetch time, counts, sha256 of the sample) so a trace manifest can cite it.

Offline: ``--from-fetched FILE...`` filters already-downloaded rows-API responses (or a previous sample)
without touching the network; tests use that path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "allenai/WildChat-1M"
ROWS_API = "https://datasets-server.huggingface.co/rows"


def rows_of(payload: Any) -> list[dict[str, Any]]:
    """Rows from a rows-API response ({"rows": [{"row": {...}}]}), a plain list of rows, or {"rows": [rows]}."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        out = []
        for item in payload["rows"]:
            if isinstance(item, dict) and isinstance(item.get("row"), dict):
                out.append(item["row"])
            elif isinstance(item, dict):
                out.append(item)
        return out
    return []


def keep(row: dict[str, Any], *, language: str, min_turns: int, model: str | None) -> bool:
    conversation = row.get("conversation")
    return (row.get("language") == language and type(row.get("turn")) is int and row["turn"] >= min_turns
            and not row.get("toxic") and not row.get("redacted") and isinstance(conversation, list)
            and len(conversation) >= 2 * min_turns and (model is None or row.get("model") == model))


def fetch_page(offset: int, length: int, *, timeout_s: float = 60.0, retries: int = 3) -> dict[str, Any]:
    query = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "train", "offset": offset, "length": length})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:  # network hiccups: retry with a pause, then give up loudly
            last = error
            time.sleep(2.0 * (attempt + 1))
    raise SystemExit(f"rows API failed at offset {offset}: {last}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True, help="sample JSON (list of raw rows)")
    ap.add_argument("--provenance", type=Path, default=None, help="default: <output>.provenance.json")
    ap.add_argument("--offset-start", type=int, default=0)
    ap.add_argument("--offset-end", type=int, default=1000, help="exclusive")
    ap.add_argument("--page", type=int, default=100, help="rows per call (API maximum 100)")
    ap.add_argument("--language", default="English")
    ap.add_argument("--min-turns", type=int, default=3)
    ap.add_argument("--model", default=None, help="keep only this source model (e.g. gpt-4-0314)")
    ap.add_argument("--from-fetched", type=Path, nargs="*", default=None,
                    help="offline: filter these rows-API responses / samples instead of fetching")
    ap.add_argument("--keep-raw-dir", type=Path, default=None, help="also store each fetched page here")
    args = ap.parse_args(argv)
    if args.page < 1 or args.page > 100:
        raise SystemExit("--page must be 1..100")
    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pages: list[dict[str, Any]] = []
    sources: list[str] = []
    if args.from_fetched is not None:
        for path in args.from_fetched:
            pages.append({"rows": rows_of(json.loads(Path(path).read_text())), "source": str(path)})
            sources.append(str(path))
    else:
        for offset in range(args.offset_start, args.offset_end, args.page):
            payload = fetch_page(offset, min(args.page, args.offset_end - offset))
            if args.keep_raw_dir is not None:
                args.keep_raw_dir.mkdir(parents=True, exist_ok=True)
                (args.keep_raw_dir / f"rows_{offset}.json").write_text(json.dumps(payload))
            pages.append({"rows": rows_of(payload), "source": f"{ROWS_API}?offset={offset}"})
            sources.append(f"offset={offset}")
            print(f"offset {offset}: {len(pages[-1]['rows'])} rows", file=sys.stderr)
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    total = 0
    for page in pages:
        for row in page["rows"]:
            total += 1
            key = str(row.get("conversation_hash") or json.dumps(row, sort_keys=True)[:200])
            if key in seen or not keep(row, language=args.language, min_turns=args.min_turns, model=args.model):
                continue
            seen.add(key)
            kept.append(row)
    kept.sort(key=lambda row: str(row.get("conversation_hash")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(kept, ensure_ascii=True, sort_keys=True)
    args.output.write_text(data)
    provenance = {
        "schema": "ws24-wildchat-sample-provenance-v1", "dataset": DATASET, "api": ROWS_API if args.from_fetched is None else "offline",
        "offsets": {"start": args.offset_start, "end": args.offset_end, "page": args.page} if args.from_fetched is None else None,
        "sources": sources, "fetched_at_utc": fetched_at, "filter": {"language": args.language, "min_turns": args.min_turns,
                                                                       "model": args.model, "toxic": False, "redacted": False},
        "rows_seen": total, "rows_kept": len(kept), "sample_path": str(args.output),
        "sample_sha256": hashlib.sha256(data.encode("ascii")).hexdigest(),
    }
    (args.provenance or args.output.with_suffix(".provenance.json")).write_text(json.dumps(provenance, indent=1, sort_keys=True))
    print(json.dumps({"rows_seen": total, "rows_kept": len(kept), "sample": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
