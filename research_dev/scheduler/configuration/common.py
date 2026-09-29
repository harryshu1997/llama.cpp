"""Typed scheduler configuration manifests: common."""

from __future__ import annotations

import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


RIG_MANIFEST_SCHEMA = "research-scheduler-rig-v1"


MODELS_MANIFEST_SCHEMA = "research-scheduler-models-v1"


EVIDENCE_MANIFEST_SCHEMA = "research-scheduler-evidence-v1"


CAMPAIGN_MANIFEST_SCHEMA = "research-scheduler-campaign-v1"


RESOLVED_CONFIGURATION_SCHEMA = "research-scheduler-resolved-config-v1"


_ENVIRONMENT_NAME = re.compile(r"S42_[A-Z0-9_]+\Z")


_SHA256 = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")


class SchedulerConfigurationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchedulerConfigurationError(message)


def _object(value: object, name: str) -> dict[str, object]:
    _require(type(value) is dict, name + " must be an object")
    return dict(value)


def _sequence(value: object, name: str) -> list[object]:
    _require(type(value) is list, name + " must be an array")
    return list(value)


def _text(value: object, name: str) -> str:
    _require(
        type(value) is str and bool(value) and value.isascii(),
        name + " must be nonempty ASCII text",
    )
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _integer(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    _require(
        type(value) is int
        and value >= minimum
        and (maximum is None or value <= maximum),
        name + " is invalid",
    )
    return value


def _boolean(value: object, name: str) -> bool:
    _require(type(value) is bool, name + " must be boolean")
    return value


def _path(value: object, base: Path, name: str) -> Path:
    raw = Path(_text(value, name))
    return raw if raw.is_absolute() else (base / raw).resolve()


def _optional_path(
    value: object, base: Path, name: str
) -> Path | None:
    return None if value is None else _path(value, base, name)


def _path_map(
    value: object, base: Path, name: str
) -> Mapping[str, Path]:
    rows = _object(value, name)
    result = {
        _text(key, name + " key"): _path(item, base, name + "." + key)
        for key, item in rows.items()
    }
    _require(result, name + " cannot be empty")
    return MappingProxyType(dict(sorted(result.items())))


def _text_map(value: object, name: str) -> Mapping[str, str]:
    rows = _object(value, name)
    result = {
        _text(key, name + " key"): _text(item, name + "." + key)
        for key, item in rows.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


def _scalar_map(
    value: object, name: str
) -> Mapping[str, int | str]:
    rows = _object(value, name)
    result: dict[str, int | str] = {}
    for key, item in rows.items():
        field = _text(key, name + " key")
        _require(
            type(item) is int
            and item >= 0
            or type(item) is str
            and bool(item)
            and item.isascii(),
            name + "." + field + " must be nonnegative integer or ASCII text",
        )
        result[field] = item
    return MappingProxyType(dict(sorted(result.items())))


def _path_json(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _decode_override(value: str, current: object) -> object:
    if type(current) is str:
        return value
    if type(current) in {bool, int, list, dict}:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise SchedulerConfigurationError(
                "environment override is not valid JSON for its target"
            ) from error
        _require(
            type(decoded) is type(current),
            "environment override changes target type",
        )
        return decoded
    raise SchedulerConfigurationError(
        "environment override target type is unsupported"
    )


def _pointer_parts(pointer: str) -> tuple[str, ...]:
    _require(
        type(pointer) is str
        and pointer.startswith("/")
        and pointer not in {"/schema", "/environment_overrides"},
        "environment override pointer is invalid",
    )
    result = tuple(
        part.replace("~1", "/").replace("~0", "~")
        for part in pointer[1:].split("/")
    )
    _require(all(result), "environment override pointer is empty")
    return result


def _pointer_target(root: object, pointer: str) -> tuple[object, str | int]:
    parts = _pointer_parts(pointer)
    current = root
    for part in parts[:-1]:
        if type(current) is dict:
            _require(part in current, "environment override path is absent")
            current = current[part]
        elif type(current) is list:
            _require(part.isdigit(), "environment override index is invalid")
            index = int(part)
            _require(index < len(current), "environment override index is absent")
            current = current[index]
        else:
            raise SchedulerConfigurationError(
                "environment override traverses a scalar"
            )
    leaf = parts[-1]
    if type(current) is list:
        _require(leaf.isdigit(), "environment override index is invalid")
        key: str | int = int(leaf)
        _require(key < len(current), "environment override index is absent")
    else:
        _require(
            type(current) is dict and leaf in current,
            "environment override target is absent",
        )
        key = leaf
    return current, key
