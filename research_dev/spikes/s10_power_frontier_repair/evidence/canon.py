#!/usr/bin/env python3
"""The single canonical-JSON and SHA-256 implementation for the evidence gate.

EVIDENCE_CONTRACT.md section 5 requires one canonical JSON and one SHA-256
implementation. Everything that hashes an evidence record, a bundle, a binding,
or a v3 instance goes through this module.

tests/test_evidence.py asserts byte-for-byte agreement between this module and
the pre-existing oracle/checker implementations on a corpus, so the frozen
temporal foundation keeps its own proven code while the evidence path provably
shares its bytes.

Integers only. A float, bool-as-int, NaN, or Infinity anywhere in the payload is
an error rather than a silently coerced value.
"""

from __future__ import annotations

import hashlib
import json

MAX_INT = 2 ** 53 - 1


class DuplicateKeyError(ValueError):
    pass


class NonAsciiError(ValueError):
    pass


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def load_strict(path):
    """Load ASCII JSON, rejecting duplicate keys and non-finite constants."""
    with open(path, encoding="ascii") as handle:
        text = handle.read()
    return loads_strict(text)


def loads_strict(text):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def canonical(value):
    """Deterministic ASCII bytes: sorted keys, no whitespace, no escapes lost."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_int(value):
    """True only for a real integer. bool is NOT an integer here."""
    return type(value) is int


def check_int(value, label):
    if not is_int(value):
        raise ValueError(f"{label} must be an integer, got {type(value).__name__}")
    if value < 0 or value > MAX_INT:
        raise ValueError(f"{label} out of declared integer range: {value}")
    return value


class MISSING:
    """Sentinel for an absent field. Never compares equal to a bound value.

    Returning None for a missing field is a fail-open bug: a binding carrying
    `value: null` would then satisfy `record[field] == value` by None == None.
    """

    def __repr__(self):
        return "<missing>"


MISSING = MISSING()


def check_integers(value, path="", allowed_bool_paths=frozenset()):
    """Recursively reject floats, booleans, and out-of-range integers.

    This is the ONE type gate, applied once at load, and it is the only thing
    that can catch this class. JSON Schema cannot: draft6+ defines `integer` as
    any number with a zero fractional part, so 900000.0 validates as an integer.
    Python then makes 900000.0 == 900000 true, so every downstream equality and
    every `>`/`<` eligibility comparison silently accepts the float.

    Booleans are rejected explicitly because bool is a subclass of int in Python,
    so True == 1 and an unwary gate reads True as a count of one.
    """
    label = path or "value"
    if isinstance(value, bool):
        if label in allowed_bool_paths:
            return
        raise ValueError(f"{label} is a boolean; booleans are not integers here")
    if isinstance(value, float):
        raise ValueError(
            f"{label} is a float ({value!r}); this contract is integer-only, and a "
            f"float would compare equal to an integer while bypassing the type gate")
    if isinstance(value, int):
        if not -MAX_INT <= value <= MAX_INT:
            raise ValueError(f"{label} is outside the declared integer range: {value}")
        return
    if value is None or isinstance(value, str):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} has a non-string key {key!r}")
            check_integers(item, f"{label}.{key}", allowed_bool_paths)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            check_integers(item, f"{label}[{index}]", allowed_bool_paths)
        return
    raise ValueError(f"{label} has unsupported type {type(value).__name__}")


def record_digest(record):
    """SHA-256 over the record with its own record_sha256 field removed."""
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return digest(body)


def bundle_digest(bundle):
    """SHA-256 over the bundle with its own bundle_sha256 field removed."""
    body = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    return digest(body)
