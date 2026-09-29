#!/usr/bin/env python3
"""Merge validated runtime observation stores without experiment policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .._internal.runtime_learning import RuntimeRouteObservationStore
from .._internal.types import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be a new absolute path")
    if any(not path.is_file() for path in args.input):
        parser.error("every input store must be a file")
    store = RuntimeRouteObservationStore()
    for index, path in enumerate(args.input):
        try:
            value = json.loads(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            parser.error("input store cannot be read: " + str(exc))
        if index == 0:
            store.import_json(value)
        else:
            store.merge_json(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        parser.error("temporary output already exists")
    try:
        temporary.write_text(
            canonical_json(store.to_json()) + "\n", encoding="ascii"
        )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(store.to_json()["store_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
