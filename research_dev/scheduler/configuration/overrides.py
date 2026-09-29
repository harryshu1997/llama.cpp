"""Typed scheduler configuration manifests: overrides."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from .._internal.types import canonical_sha256
from .common import (
    SchedulerConfigurationError,
    _ENVIRONMENT_NAME,
    _decode_override,
    _object,
    _pointer_target,
    _require,
    _text,
)


@dataclass(frozen=True)
class ConfigurationOverride:
    environment_name: str
    manifest_schema: str
    pointer: str
    previous_value: object
    resolved_value: object

    def to_json(self) -> dict[str, object]:
        return {
            "environment_name": self.environment_name,
            "manifest_schema": self.manifest_schema,
            "pointer": self.pointer,
            "previous_value": self.previous_value,
            "resolved_value": self.resolved_value,
        }


@dataclass(frozen=True)
class ManifestIdentity:
    path: Path
    schema: str
    source_sha256: str
    resolved_sha256: str

    def to_json(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "resolved_sha256": self.resolved_sha256,
            "schema": self.schema,
            "source_sha256": self.source_sha256,
        }


def _load_raw_manifest(
    path: Path,
    schema: str,
    environ: Mapping[str, str],
) -> tuple[dict[str, object], str, tuple[ConfigurationOverride, ...]]:
    _require(path.is_absolute() and path.is_file(), "manifest path is invalid")
    try:
        source = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SchedulerConfigurationError("manifest cannot be loaded") from error
    raw = _object(source, "manifest")
    _require(raw.get("schema") == schema, "manifest schema differs")
    source_sha256 = canonical_sha256(raw)
    declarations = raw.get("environment_overrides", {})
    declarations = _object(declarations, "environment overrides")
    resolved = deepcopy(raw)
    overrides = []
    for environment_name, pointer_value in sorted(declarations.items()):
        name = _text(environment_name, "environment override name")
        pointer = _text(pointer_value, "environment override pointer")
        _require(
            _ENVIRONMENT_NAME.fullmatch(name) is not None,
            "environment override name is not S42_*",
        )
        if name not in environ:
            continue
        parent, key = _pointer_target(resolved, pointer)
        previous = parent[key]
        current = _decode_override(environ[name], previous)
        parent[key] = current
        overrides.append(ConfigurationOverride(
            environment_name=name,
            manifest_schema=schema,
            pointer=pointer,
            previous_value=previous,
            resolved_value=current,
        ))
    return resolved, source_sha256, tuple(overrides)
