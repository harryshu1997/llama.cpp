#!/usr/bin/env python3
"""Normalize learning keys from validated physical result receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .._internal.runtime_learning import RuntimeRouteObservationStore
from .._internal.types import canonical_json, canonical_sha256


def _contexts(value: object) -> tuple[tuple[str, dict[str, int]], ...]:
    if type(value) is not dict or type(value.get("request_results")) is not list:
        raise ValueError("physical result request rows are invalid")
    result = []
    for row in value["request_results"]:
        if type(row) is not dict:
            raise ValueError("physical result request row is invalid")
        completion = row.get("completion")
        ticket = row.get("terminal_ticket")
        if (
            type(completion) is not dict
            or type(ticket) is not dict
            or type(completion.get("execution_receipt")) is not dict
            or completion["execution_receipt"].get("status") != "COMPLETED"
            or completion["execution_receipt"].get("ticket_id")
                != ticket.get("ticket_id")
            or type(ticket.get("runtime_observation")) is not dict
            or type(
                ticket["runtime_observation"].get("cost_features")
            ) is not dict
        ):
            raise ValueError("physical result completion is invalid")
        features = ticket["runtime_observation"]["cost_features"]
        if any(
            type(name) is not str
            or not name
            or not name.isascii()
            or type(feature) is not int
            for name, feature in features.items()
        ):
            raise ValueError("physical result cost features are invalid")
        result.append((
            canonical_sha256(completion["execution_receipt"]),
            dict(features),
        ))
    if not result:
        raise ValueError("physical result has no completed requests")
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("input observation store must be a file")
    if any(not path.is_file() for path in args.result):
        parser.error("every physical result must be a file")
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be a new absolute path")
    try:
        store_value = json.loads(args.input.read_text(encoding="ascii"))
        contexts = tuple(
            context
            for path in args.result
            for context in _contexts(json.loads(
                path.read_text(encoding="ascii")
            ))
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error("physical learning evidence is invalid: " + str(exc))
    store = RuntimeRouteObservationStore()
    store.import_json(store_value)
    store.normalize_template_features(contexts)
    output = store.to_json()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        parser.error("temporary output already exists")
    try:
        temporary.write_text(
            canonical_json(output) + "\n", encoding="ascii"
        )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output["store_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
