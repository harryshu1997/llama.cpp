#!/usr/bin/env python3
"""Load the reviewed E2 modules from the exact source bytes E2A pins."""

from __future__ import annotations

import hashlib
import pathlib
import sys
import types

import e2a_canon as pinned_canon


E2_SRC = (pathlib.Path(__file__).resolve().parents[2]
          / "s10_matched_energy_e2" / "src")

PINNED_SOURCES = {
    "e2_canon": (E2_SRC / "e2_canon.py",
                 "977ce31503d41113015b0ab097845bbe11e93b27f222d0d18a8ac8461eccea89"),
    "integrator": (E2_SRC / "integrator.py",
                   "d02b35a19f74feab0efc43e21ae5d5b0af6a33844375e18ebcc3339ac13cb38c"),
    "comparator": (E2_SRC / "comparator.py",
                   "3a230606ecc366a090ccfbd2910ecd1529e84180ba768eb89c8715dfe9164b00"),
}


class E2PinError(RuntimeError):
    """The E2 implementation on disk is not the reviewed implementation."""


def _source(name):
    path, expected = PINNED_SOURCES[name]
    if path.is_symlink() or not path.is_file():
        raise E2PinError(f"E_E2_UNPINNED: {path} is not a regular source file")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise E2PinError(
            f"E_E2_UNPINNED: {path} hashes to {actual}, expected {expected}")
    return path, data


def _execute(name, dependencies):
    path, data = _source(name)
    module = types.ModuleType(f"_s10_e2a_pinned_{name}")
    module.__file__ = str(path)
    module.__package__ = ""

    saved = {key: sys.modules.get(key) for key in dependencies}
    try:
        sys.modules.update(dependencies)
        exec(compile(data, str(path), "exec"), module.__dict__)
    except Exception as exc:
        raise E2PinError(
            f"E_E2_UNPINNED: pinned {name} source does not execute: "
            f"{type(exc).__name__}: {exc}") from exc
    finally:
        for key, previous in saved.items():
            if previous is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous
    return module


def _build_e2_canon_adapter():
    """Expose E2's canon API without executing its transitive pyc loader.

    The E2 wrapper is still pinned by digest. Its implementation delegates to
    E1 through SourceFileLoader.exec_module(), though, so executing it would
    allow a timestamp-valid poisoned E1 .pyc to replace the source bytes E2A
    reviewed. E2A already loads that exact E1 source through compile(data).
    Adapt that verified module to E2's small wrapper API instead.
    """
    path, _data = _source("e2_canon")
    module = types.ModuleType("_s10_e2a_pinned_e2_canon_adapter")
    module.__file__ = str(path)
    for name in (
            "MAX_INT", "MISSING", "DuplicateKeyError", "canonical", "digest",
            "load_strict", "loads_strict", "is_int", "check_int",
            "check_integers", "file_sha256"):
        setattr(module, name, getattr(pinned_canon, name))

    def record_digest(record):
        body = {key: value for key, value in record.items()
                if key != "record_sha256"}
        return module.digest(body)

    def seal(record):
        record["record_sha256"] = record_digest(record)
        return record

    def sha256_bytes(data):
        return hashlib.sha256(data).hexdigest()

    module.record_digest = record_digest
    module.seal = seal
    module.sha256_bytes = sha256_bytes
    return module


e2_canon = _build_e2_canon_adapter()
integrator = _execute("integrator", {"e2_canon": e2_canon})
comparator = _execute("comparator", {
    "e2_canon": e2_canon,
    "integrator": integrator,
})
