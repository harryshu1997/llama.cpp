#!/usr/bin/env python3
"""Canonical JSON, strict loading, and the integer type gate for E2.

REUSE, NOT COPY: E1's `evidence/canon.py` is imported read-only so both gates
agree byte-for-byte on canonical bytes and SHA-256. Nothing here mutates E1, and
E2 imports NO E1 decision logic (no binder, no boundary, no validator) -- see
CONTRACT.md section 0. tests/test_e2.py asserts that separation by inspecting
sys.modules after an E2 run.

The type gate is E1's hard-won lesson, restated because it is the only thing that
catches this class: JSON Schema draft6+ defines `integer` as any number with a
zero fractional part, so 900000.0 validates as an integer; Python then makes
900000.0 == 900000, so a float satisfies every equality AND every ordering
comparison. bool is a subclass of int, so True == 1. Only type(x) is int catches
either, and it must run BEFORE any comparison.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib

E1_EVIDENCE = (pathlib.Path(__file__).resolve().parents[2]
               / "s10_power_frontier_repair" / "evidence")
if not E1_EVIDENCE.is_dir():
    raise RuntimeError(f"E1 evidence directory not found at {E1_EVIDENCE}")
CANON_PATH = E1_EVIDENCE / "canon.py"
_spec = importlib.util.spec_from_file_location("_s10_e1_evidence_canon", CANON_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load E1 canonicalization module at {CANON_PATH}")
_e1_canon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e1_canon)

MAX_INT = _e1_canon.MAX_INT
MISSING = _e1_canon.MISSING
DuplicateKeyError = _e1_canon.DuplicateKeyError

canonical = _e1_canon.canonical
digest = _e1_canon.digest
load_strict = _e1_canon.load_strict
loads_strict = _e1_canon.loads_strict
is_int = _e1_canon.is_int
check_int = _e1_canon.check_int
check_integers = _e1_canon.check_integers
file_sha256 = _e1_canon.file_sha256


def record_digest(record):
    """SHA-256 over the record with its own record_sha256 removed."""
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return digest(body)


def seal(record):
    record["record_sha256"] = record_digest(record)
    return record


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()
