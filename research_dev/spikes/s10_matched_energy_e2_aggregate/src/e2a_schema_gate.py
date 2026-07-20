#!/usr/bin/python3
"""Draft-2020 JSON Schema worker for E2A.

Runs under the system Python that owns /usr/bin/jsonschema, invoked by the parent
as `/usr/bin/python3 -I e2a_schema_gate.py`. The `-I` flag is load-bearing: it
isolates the worker from PYTHONPATH, so a hostile `jsonschema.py` on the path
cannot replace the engine with one that approves everything.

Refuses any non-local reference keyword, not just `$ref`: `$dynamicRef`,
`$recursiveRef`, and a remote `$id` can each pull in an outside definition at
validation time.
"""

from __future__ import annotations

import json
import pathlib
import sys

try:
    from jsonschema import Draft202012Validator
except Exception as exc:  # pragma: no cover - exercised by the parent refusal
    print(json.dumps({"engine_error": f"cannot import jsonschema: {exc}"}))
    sys.exit(2)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_NAMES = {
    "plan": "pre_run_plan",
    "plan_anchor": "plan_anchor_receipt",
    "ledger_close": "ledger_close_receipt",
    "request_set": "request_set_manifest",
    "request_outcomes": "request_outcome_record",
    "ledger": "attempt_ledger",
    "lifecycle": "lifecycle_record",
    "aggregate": "aggregate_comparison",
    "bundle": "bundle",
    "route": "route_schedule",
}

REF_KEYS = ("$ref", "$dynamicRef", "$recursiveRef")


def _pointer(parts):
    if not parts:
        return "/"
    escaped = (str(part).replace("~", "~0").replace("/", "~1") for part in parts)
    return "/" + "/".join(escaped)


def _assert_local_refs(value, depth=0):
    if depth > 64:
        raise ValueError("schema nests too deeply")
    if isinstance(value, dict):
        for key, item in value.items():
            if key in REF_KEYS:
                if not isinstance(item, str) or not item.startswith("#/") \
                        or "//" in item:
                    raise ValueError(f"non-local schema reference "
                                     f"{key}={item!r}")
            if key == "$id" and depth > 0:
                raise ValueError("nested $id can rebase reference resolution")
            _assert_local_refs(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _assert_local_refs(item, depth + 1)


def main():
    try:
        request = json.load(sys.stdin)
        if set(request) != {"kind", "document"}:
            raise ValueError("request must contain exactly kind and document")
        kind = request["kind"]
        if kind not in SCHEMA_NAMES:
            raise ValueError(f"unknown document kind {kind!r}")
        document = request["document"]
        if not isinstance(document, dict):
            raise ValueError("document must be an object")
        version = document.get("schema_version")
        if type(version) is not int or version not in (1, 2, 3, 4):
            raise ValueError(f"unsupported schema_version {version!r}")
        schema_path = (ROOT / "schemas" /
                       f"{SCHEMA_NAMES[kind]}.v{version}.schema.json")
        with schema_path.open(encoding="ascii") as handle:
            schema = json.load(handle)
        _assert_local_refs(schema)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        errors = sorted(
            validator.iter_errors(document),
            key=lambda err: (tuple(str(part) for part in err.absolute_path),
                             str(err.validator), err.message),
        )
        print(json.dumps({
            "errors": [
                {"path": _pointer(error.absolute_path),
                 "validator": str(error.validator),
                 "message": error.message}
                for error in errors
            ]
        }, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"engine_error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
