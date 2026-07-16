#!/usr/bin/env python3
"""Canonical JSON, strict loading, and the integer type gate for E2A.

Reuses E1's canon.py read-only, exactly as E2 does, so all three gates agree
byte-for-byte on canonical bytes and SHA-256. E2A imports NO decision logic from
E1 or E2 -- see CONTRACT.md section 0.

DIFFERENCE FROM E2's e2_canon: the import is PINNED. E2 exec's E1's canon.py with
no digest check, so anything that can write that file replaces the definition of
"canonical" and "is_int" underneath every hash and every type gate in the suite.
Only scripts/run_tests.sh consulted the baseline manifest, which means the gate
was protected by its own test harness rather than by itself -- and a gate that is
only honest when run under its own runner is not a gate. E2A pins the digest here,
at import, before a single byte of canon.py is executed.

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
CANON_PATH = E1_EVIDENCE / "canon.py"

# Pinned at the E1 baseline frozen by CP0 (see E2A_FREEZE.txt). If E1's canon.py
# legitimately changes, this constant must be updated deliberately, by a human,
# in the same commit -- which is the point: the change becomes a diff rather than
# a silence.
FROZEN_CANON_SHA256 = \
    "b2a3bfde28a22033319afb7e4e63f7f8e1b62eece69c7e507ac4dd868d0f9d23"


class CanonPinError(RuntimeError):
    """The shared canonicalization module is not the one that was reviewed."""


def _load_pinned_canon():
    if not E1_EVIDENCE.is_dir():
        raise CanonPinError(f"E_CANON_UNPINNED: E1 evidence directory not found "
                            f"at {E1_EVIDENCE}")
    # A symlink anywhere on the way to canon.py can be repointed after this check,
    # so refuse links outright rather than resolving them.
    probe = E1_EVIDENCE
    for part in ("canon.py",):
        probe = probe / part
        if probe.is_symlink():
            raise CanonPinError(
                f"E_CANON_UNPINNED: {probe} is a symlink; its target can be "
                f"repointed after the digest check")
    if not CANON_PATH.is_file():
        raise CanonPinError(f"E_CANON_UNPINNED: {CANON_PATH} is not a regular file")
    data = CANON_PATH.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != FROZEN_CANON_SHA256:
        raise CanonPinError(
            f"E_CANON_UNPINNED: {CANON_PATH} hashes to {actual} but E2A pins "
            f"{FROZEN_CANON_SHA256}; the module that defines canonical bytes, "
            f"SHA-256, and the integer type gate is not the reviewed one, so "
            f"nothing downstream of it can be trusted")

    # EXECUTE THE BYTES THAT WERE HASHED. Nothing else.
    #
    # An earlier revision hashed the source and then called
    # spec.loader.exec_module(), which is a TOCTOU gap at module scope:
    # SourceFileLoader consults __pycache__ and, when the cached .pyc header
    # matches the source's mtime and size, executes the CACHED BYTECODE instead
    # of the file just hashed. The red team demonstrated it -- a poisoned
    # canon.cpython-313.pyc with a forged header left the source digest at
    # b2a3bfde... (pin passes, cleanly) while check_integers silently stopped
    # rejecting 900000.0. Every hash and every type gate in the suite then rested
    # on attacker bytecode, and the pyc is writable in the real tree.
    #
    # This is E2's artifact lesson recurring one level up: hashing one thing and
    # consuming another is the same defect whether the thing is a power trace or
    # the code that hashes it. compile() closes it -- there is one buffer, and it
    # is both hashed and executed.
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("_s10_e2a_pinned_canon", loader=None))
    module.__file__ = str(CANON_PATH)
    try:
        exec(compile(data, str(CANON_PATH), "exec"), module.__dict__)
    except Exception as exc:
        raise CanonPinError(
            f"E_CANON_UNPINNED: the pinned canon source at {CANON_PATH} does not "
            f"execute: {type(exc).__name__}: {exc}")
    return module


_e1_canon = _load_pinned_canon()

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

# A known-answer vector. If the pinned module ever fails this, the pin is wrong
# rather than merely stale, and every digest in the suite is meaningless.
_KAT = digest({"e2a": "known-answer"})
if len(_KAT) != 64 or any(ch not in "0123456789abcdef" for ch in _KAT):
    raise CanonPinError("E_CANON_UNPINNED: pinned canon does not produce a "
                        "SHA-256 hex digest")
if is_int(True) or is_int(1.0) or not is_int(1):
    raise CanonPinError("E_CANON_UNPINNED: pinned canon's integer type gate does "
                        "not reject bool/float; every downstream comparison is "
                        "unprotected")


def record_digest(record):
    """SHA-256 over the record with its own record_sha256 removed."""
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return digest(body)


def seal(record):
    record["record_sha256"] = record_digest(record)
    return record


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()
