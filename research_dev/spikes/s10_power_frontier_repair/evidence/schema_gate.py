#!/usr/bin/python3
"""Draft-2020 JSON Schema gate for the S10 v3 evidence path.

The repository's ambient Python does not provide jsonschema. This small worker
runs with the system Python that owns /usr/bin/jsonschema, so the live binder
can enforce the same schemas without importing packages across Python versions.
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
SCHEMAS = {
    "bundle": ROOT / "schemas" / "evidence_bundle.v3.schema.json",
    "instance": ROOT / "schemas" / "instance.v3.schema.json",
    "certificate": ROOT / "schemas" / "certificate.v3.schema.json",
}


def _pointer(parts):
    if not parts:
        return "/"
    escaped = (str(part).replace("~", "~0").replace("/", "~1")
               for part in parts)
    return "/" + "/".join(escaped)


def _assert_local_refs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$ref" and (not isinstance(item, str) or
                                   not item.startswith("#/") or
                                   "//" in item):
                raise ValueError(f"non-local schema reference {item!r}")
            _assert_local_refs(item)
    elif isinstance(value, list):
        for item in value:
            _assert_local_refs(item)


def main():
    try:
        request = json.load(sys.stdin)
        if set(request) != {"kind", "document"}:
            raise ValueError("request must contain exactly kind and document")
        kind = request["kind"]
        if kind not in SCHEMAS:
            raise ValueError(f"unknown document kind {kind!r}")
        with SCHEMAS[kind].open(encoding="ascii") as handle:
            schema = json.load(handle)
        _assert_local_refs(schema)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        errors = sorted(
            validator.iter_errors(request["document"]),
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
