#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


MAX_INT = (1 << 63) - 1
MAX_JSON_BYTES = 64 * 1024 * 1024
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)


class EvidenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    actual = set(value)
    require(
        actual == expected,
        f"E_KEYS: {field}: missing={sorted(expected - actual)}, "
        f"unknown={sorted(actual - expected)}",
    )
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int, f"E_TYPE: {field}")
    require(minimum <= value <= MAX_INT, f"E_RANGE: {field}")
    return value


def text(value: Any, field: str, maximum: int = 32768) -> str:
    require(
        type(value) is str and 0 < len(value) <= maximum,
        f"E_TYPE: {field}",
    )
    require(
        value.isascii() and "\x00" not in value and "\n" not in value,
        f"E_ASCII: {field}",
    )
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field, 64)
    require(DIGEST_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def uuid(value: Any, field: str) -> str:
    value = text(value, field, 64)
    require(UUID_RE.fullmatch(value) is not None, f"E_UUID: {field}")
    return value


def absolute_path(value: Any, field: str) -> str:
    value = text(value, field)
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, f"E_PATH: {field}")
    return value


def relative_path(value: Any, field: str) -> str:
    value = text(value, field)
    path = Path(value)
    require(
        not path.is_absolute()
        and value not in (".", "..")
        and ".." not in path.parts,
        f"E_PATH: {field}",
    )
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
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


def canonical_compact(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise EvidenceError("E_CANONICAL") from error


def parse_json(raw: bytes, field: str) -> Any:
    require(len(raw) <= MAX_JSON_BYTES, f"E_SIZE: {field}")
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"E_JSON: {field}") from error


def read_regular(path: Path, maximum: int = MAX_JSON_BYTES) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise EvidenceError(f"E_READ: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_FILE_TYPE: {path}")
        require(0 < before.st_size <= maximum, f"E_FILE_SIZE: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
            require(len(raw) <= maximum, f"E_FILE_SIZE: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )
    require(identity(before) == identity(after), f"E_FILE_CHANGED: {path}")
    require(len(raw) == before.st_size, f"E_FILE_CHANGED: {path}")
    return bytes(raw)


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path)
    value = parse_json(raw, str(path))
    require(type(value) is dict, f"E_TYPE: {path}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {path}")
    return value, raw


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(read_regular(path, 512 * 1024 * 1024))


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
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return raw
