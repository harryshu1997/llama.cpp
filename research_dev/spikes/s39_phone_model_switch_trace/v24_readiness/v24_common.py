#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


MAX_INT = (1 << 63) - 1
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


class EvidenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"E_JSON_NUMBER: {value}")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise EvidenceError("E_CANONICAL") from error


def parse_json(raw: bytes, field: str) -> Any:
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"E_JSON: {field}: {error}") from error


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise EvidenceError(f"E_READ: {path}: {error}") from error
    value = parse_json(raw, str(path))
    require(type(value) is dict, f"E_TYPE: {path}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {path}")
    return value, raw


def write_exclusive(path: Path, value: Any) -> bytes:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return raw


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest_value = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest_value.update(block)
    except OSError as error:
        raise EvidenceError(f"E_READ: {path}: {error}") from error
    return digest_value.hexdigest()


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}: expected {expected!r}, got {value!r}",
    )


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    actual = set(value)
    require(
        actual == keys,
        f"E_KEYS: {field}: missing={sorted(keys - actual)}, "
        f"unknown={sorted(actual - keys)}",
    )
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int, f"E_TYPE: {field}")
    require(minimum <= value <= MAX_INT, f"E_RANGE: {field}")
    return value


def text(value: Any, field: str) -> str:
    require(type(value) is str and bool(value), f"E_TYPE: {field}")
    require(value.isascii(), f"E_ASCII: {field}")
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field)
    require(DIGEST_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def uuid(value: Any, field: str) -> str:
    value = text(value, field)
    require(UUID_RE.fullmatch(value) is not None, f"E_UUID: {field}")
    return value


def absolute_path(value: Any, field: str) -> str:
    value = text(value, field)
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, f"E_PATH: {field}")
    return value


def stat_record(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(
        value,
        {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
        field,
    )
    for key in ("ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"):
        integer(value[key], f"{field}.{key}")
    require(value["inode"] > 0 and value["size"] > 0, f"E_STAT: {field}")
    return value


def component_key(endpoint: str, path: str) -> str:
    return f"{endpoint}\0{path}"
