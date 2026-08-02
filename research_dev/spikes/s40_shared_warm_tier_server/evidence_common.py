#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


SHA256_RE = re.compile(r"[0-9a-f]{64}")


class EvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_int(value: Any, field: str, minimum: int = 0) -> int:
    require(is_int(value), f"{field}: expected integer")
    require(value >= minimum, f"{field}: expected >= {minimum}")
    return value


def require_string(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(value), f"{field}: expected string")
    return value


def reject_constant(value: str) -> None:
    raise EvidenceError(f"invalid JSON constant {value}")


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def parse_json(raw: bytes, field: str) -> Any:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{field}: expected ASCII") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise EvidenceError(f"{field}: invalid JSON: {error}") from error


def read_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = parse_json(path.read_bytes(), field)
    except OSError as error:
        raise EvidenceError(f"{field}: cannot read {path}: {error}") from error
    require(isinstance(value, dict), f"{field}: expected object")
    return value


def read_jsonl(path: Path, field: str) -> list[dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise EvidenceError(f"{field}: cannot read {path}: {error}") from error
    require(raw.endswith(b"\n"), f"{field}: missing final newline")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines()):
        value = parse_json(line, f"{field}[{index}]")
        require(isinstance(value, dict), f"{field}[{index}]: expected object")
        rows.append(value)
    return rows


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise EvidenceError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def validate_digest(value: Any, field: str) -> str:
    digest = require_string(value, field)
    require(SHA256_RE.fullmatch(digest) is not None, f"{field}: invalid SHA-256")
    return digest


def percentile(values: list[int], numerator: int, denominator: int) -> int:
    require(bool(values), "percentile: empty input")
    require(
        is_int(numerator) and is_int(denominator)
        and 0 <= numerator <= denominator and denominator > 0,
        "percentile: invalid fraction",
    )
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[max(0, rank - 1)]
