#!/usr/bin/env python3
"""Shared primitives for the S14 CP0-c frozen island catalog.

Canonical JSON and sha256 match the frozen S8 house convention
(normalize_trace.canonical_json / sha256_bytes): sort_keys, compact separators,
ensure_ascii, and a "sha256:" prefix. They are vendored here (about 20 lines)
so the freeze artifact is self-contained and auditable, exactly as the schema
files declare themselves self-contained.

The build_catalog and validate_catalog modules both import these primitives, but
the meaningful independence between them is that the validator RE-DERIVES and
RE-CHECKS every value the builder asserts (content-address hashes, cross-refs,
boundary bounds, fail-closed verdict), rather than sharing construction logic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class CatalogError(Exception):
    """Raised on any fail-closed catalog violation."""


def canonical_json(value: Any) -> bytes:
    """Deterministic canonical serialization (matches S8 normalize_trace)."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_of(value: Any) -> str:
    """sha256 of the canonical serialization of a JSON value."""
    return sha256_bytes(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: str | Path) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise CatalogError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    def reject_nonfinite(token: str) -> Any:
        raise CatalogError(f"non-finite JSON constant: {token}")

    with Path(path).open("rb") as source:
        return json.loads(
            source.read().decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )


# Content-address derivations. graph_hash and weight_set_id are v0 DERIVED
# identities: deterministic functions of (model_version, layer_range,
# attention_class). They satisfy the atlas requirement that a different layer
# range or attention class is a different identity, and they are stable across
# reruns. They are NOT yet the compiled ggml graph digest or the S9 prepared
# image digest; CATALOG.md records that gap and CP1 replaces them with the real
# test-export-graph-ops graph hash and S9 model-manifest weight digest.

def derived_graph_hash(model_version: str, layer_range: dict[str, int], attention_class: str) -> str:
    return sha256_of(
        {
            "role": "graph_hash_v0_derived",
            "model_version": model_version,
            "layer_range": {
                "start": layer_range["start"],
                "end": layer_range["end"],
                "n_layer_total": layer_range["n_layer_total"],
            },
            "attention_class": attention_class,
        }
    )


def derived_weight_set_id(model_version: str, layer_range: dict[str, int]) -> str:
    return sha256_of(
        {
            "role": "weight_set_id_v0_derived",
            "model_version": model_version,
            "layer_range": {
                "start": layer_range["start"],
                "end": layer_range["end"],
                "n_layer_total": layer_range["n_layer_total"],
            },
        }
    )


def descriptor_hash(descriptor: dict[str, Any]) -> str:
    """Content address of an island descriptor: sha256 over the canonical
    descriptor with the descriptor_hash field itself removed."""
    body = {key: value for key, value in descriptor.items() if key != "descriptor_hash"}
    return sha256_of(body)
