"""Shared validation and artifact I/O for BurstGPT campaigns."""

import hashlib
import json
from pathlib import Path
from typing import Any


CONFIRMATION = "RUN_UNIFIED_FP16_LLAMA_OVERLAY"


class UnifiedTraceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise UnifiedTraceError(message)


def canonical(value: object) -> bytes:
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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(), 1):
        row = json.loads(raw)
        require(type(row) is dict, f"trace row {line_number}")
        rows.append(row)
    require(rows, f"trace is empty: {path}")
    return rows
