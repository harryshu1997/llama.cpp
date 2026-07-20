#!/usr/bin/env python3
"""Deterministic Gate-A normalizer for the pinned S8 trace sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, BinaryIO, Iterator


SAFE_MAX = 2**53 - 1
WINDOW_US = 900_000_000
SCENARIOS = ("low", "median", "high", "burst")
QUANTILES = {
    "low": (1, 10),
    "median": (1, 2),
    "high": (9, 10),
}
NORMALIZER_CODE_FILES = ("normalize_trace.py",)
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _is_direct_source_execution() -> bool:
    try:
        module_path = Path(__file__).resolve()
        argv_path = Path(sys.argv[0]).resolve()
        prefix = module_path.read_bytes()[: len(importlib.util.MAGIC_NUMBER)]
    except OSError:
        return False
    return (
        __name__ == "__main__"
        and __spec__ is None
        and globals().get("__cached__") is None
        and module_path.suffix == ".py"
        and argv_path == module_path
        and prefix != importlib.util.MAGIC_NUMBER
    )


DIRECT_SOURCE_EXECUTION = _is_direct_source_execution()


class NormalizeError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def fail(code: str, message: str) -> None:
    raise NormalizeError(code, message)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _require_int(
    value: Any,
    path: str,
    minimum: int = 0,
    maximum: int = SAFE_MAX,
) -> int:
    if not _is_int(value) or not minimum <= value <= maximum:
        fail("E_CONFIG", f"{path} must be an integer in [{minimum},{maximum}]")
    return value


def _require_string(value: Any, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        fail("E_CONFIG", f"{path} must be a non-empty string")
    return value


def _exact_keys(
    value: Any,
    required: set[str],
    path: str,
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail("E_CONFIG", f"{path} must be an object")
    allowed = required | (optional or set())
    missing = sorted(required - set(value))
    extra = sorted(set(value) - allowed)
    if missing or extra:
        fail("E_CONFIG", f"{path} keys mismatch: missing={missing}, extra={extra}")
    return value


def _reject_constant(value: str) -> None:
    fail("E_JSON", f"non-finite JSON number is forbidden: {value}")


def _parse_json_integer(value: str) -> int:
    if re.fullmatch(r"^[0-9]+$", value) is None:
        fail("E_INTEGER", f"signed JSON integer is forbidden: {value}")
    return int(value)


def _reject_json_float(value: str) -> None:
    fail("E_JSON", f"JSON float is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("E_JSON_DUPLICATE_KEY", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_strict_bytes(data: bytes, label: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail("E_UTF8", f"{label}: invalid UTF-8 at byte {exc.start}")
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_parse_json_integer,
            parse_float=_reject_json_float,
            parse_constant=_reject_constant,
        )
    except NormalizeError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        fail("E_JSON", f"{label}: invalid JSON: {exc}")


def load_json_strict(path: Path) -> Any:
    try:
        return load_json_strict_bytes(path.read_bytes(), str(path))
    except OSError as exc:
        fail("E_IO", f"cannot read {path}: {exc}")


def canonical_json(value: Any) -> bytes:
    _reject_unsupported_json(value, "$")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        fail("E_CANONICAL", f"cannot serialize canonical JSON: {exc}")


def _reject_unsupported_json(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool)) or _is_int(value):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_unsupported_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                fail("E_CANONICAL", f"{path}: JSON object key is not a string")
            _reject_unsupported_json(item, f"{path}.{key}")
        return
    fail("E_CANONICAL", f"{path}: unsupported JSON value type {type(value).__name__}")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        fail("E_IO", f"cannot hash {path}: {exc}")
    return "sha256:" + digest.hexdigest()


def normalizer_version() -> str:
    here = Path(__file__).resolve().parent
    lines = []
    for relpath in sorted(NORMALIZER_CODE_FILES):
        digest = sha256_file(here / relpath).removeprefix("sha256:")
        lines.append(f"{relpath}  {digest}")
    return sha256_bytes("\n".join(lines).encode("ascii"))


STARTUP_NORMALIZER_VERSION = (
    normalizer_version()
    if DIRECT_SOURCE_EXECUTION or __name__ != "__main__"
    else "sha256:" + "0" * 64
)


def _compile_grammar(pattern: Any, path: str) -> re.Pattern[str]:
    _require_string(pattern, path)
    try:
        return re.compile(pattern)
    except re.error as exc:
        fail("E_CONFIG", f"{path} is not a valid regular expression: {exc}")


def validate_config(config: Any) -> dict[str, Any]:
    top = _exact_keys(
        config,
        {
            "schema_version",
            "config_version",
            "source",
            "provenance",
            "origin",
            "format",
            "encoding",
            "bom",
            "newline",
            "timestamp",
            "gate_a",
            "mapping",
        },
        "$",
        {"csv", "jsonl"},
    )
    if top["schema_version"] != 1:
        fail("E_CONFIG", "schema_version must equal 1")
    _require_string(top["config_version"], "$.config_version")
    source = _require_string(top["source"], "$.source")
    if not SAFE_NAME_RE.fullmatch(source):
        fail("E_CONFIG", "$.source is not a safe output-name component")
    if top["provenance"] not in ("real", "real_decomposed"):
        fail("E_CONFIG", "$.provenance is unsupported")
    if top["format"] not in ("csv", "jsonl"):
        fail("E_CONFIG", "$.format is unsupported")
    supported_pair = {
        ("csv", "burstgpt-v2", "burstgpt-v2.0-cfg-1"),
        ("jsonl", "ragpulse", "ragpulse-cfg-1"),
    }
    if (top["format"], source, top["config_version"]) not in supported_pair:
        fail("E_CONFIG", "only the two pinned S8 source configurations are supported")
    if top["encoding"] != "utf-8" or top["bom"] != "reject":
        fail("E_CONFIG", "only UTF-8 without BOM is supported")
    if top["newline"] != "lf_or_crlf":
        fail("E_CONFIG", "only lf_or_crlf newlines are supported")

    origin = _exact_keys(
        top["origin"],
        {
            "url",
            "revision",
            "filename",
            "bytes",
            "sha256",
            "line_count",
            "record_count",
            "license",
        },
        "$.origin",
    )
    for key in ("url", "revision", "filename", "license"):
        _require_string(origin[key], f"$.origin.{key}")
    _require_int(origin["bytes"], "$.origin.bytes", 1)
    _require_int(origin["line_count"], "$.origin.line_count", 1)
    _require_int(origin["record_count"], "$.origin.record_count", 1)
    if not isinstance(origin["sha256"], str) or not SHA256_RE.fullmatch(origin["sha256"]):
        fail("E_CONFIG", "$.origin.sha256 must be sha256:<64 lowercase hex>")

    timestamp = _exact_keys(
        top["timestamp"],
        {"field", "grammar", "unit", "to_us", "policy", "overflow_max"},
        "$.timestamp",
    )
    _require_string(timestamp["field"], "$.timestamp.field")
    _compile_grammar(timestamp["grammar"], "$.timestamp.grammar")
    if timestamp["unit"] != "seconds":
        fail("E_CONFIG", "$.timestamp.unit must equal seconds")
    if timestamp["to_us"] != "decimal_round_half_up_1e6":
        fail("E_CONFIG", "$.timestamp.to_us is unsupported")
    if timestamp["policy"] not in ("require_nondecreasing", "sort_stable"):
        fail("E_CONFIG", "$.timestamp.policy is unsupported")
    if timestamp["overflow_max"] != SAFE_MAX:
        fail("E_CONFIG", f"$.timestamp.overflow_max must equal {SAFE_MAX}")

    gate_a = _exact_keys(
        top["gate_a"],
        {
            "deadline_us",
            "deadline_provenance",
            "priority_class",
            "priority_provenance",
            "observed_latency_us",
            "images",
            "audio_ms",
        },
        "$.gate_a",
    )
    expected_gate = {
        "deadline_us": None,
        "deadline_provenance": "none",
        "priority_class": None,
        "priority_provenance": "none",
        "observed_latency_us": None,
        "images": 0,
        "audio_ms": 0,
    }
    if gate_a != expected_gate:
        fail("E_CONFIG", "$.gate_a does not match the frozen Gate-A constants")

    mapping = _exact_keys(
        top["mapping"],
        {
            "t_from",
            "input_tokens",
            "output_tokens",
            "session_id",
            "service",
            "model_class",
            "retrieved_chunks",
            "cache_keys",
            "source_fields",
        },
        "$.mapping",
    )
    if mapping["t_from"] != timestamp["field"]:
        fail("E_CONFIG", "$.mapping.t_from must equal $.timestamp.field")
    for name in ("input_tokens", "output_tokens"):
        spec = _exact_keys(mapping[name], {"from"}, f"$.mapping.{name}", {"grammar"})
        _require_string(spec["from"], f"$.mapping.{name}.from")
        if "grammar" in spec:
            _compile_grammar(spec["grammar"], f"$.mapping.{name}.grammar")
    session = _exact_keys(
        mapping["session_id"],
        {"from", "blank_to_null"},
        "$.mapping.session_id",
    )
    _require_string(session["from"], "$.mapping.session_id.from")
    if type(session["blank_to_null"]) is not bool:
        fail("E_CONFIG", "$.mapping.session_id.blank_to_null must be boolean")
    _validate_value_mapping(mapping["service"], "$.mapping.service")
    _validate_value_mapping(mapping["model_class"], "$.mapping.model_class")
    _validate_retrieved_chunks(mapping["retrieved_chunks"])
    _validate_cache_keys(mapping["cache_keys"])
    _validate_source_fields(mapping["source_fields"])

    if top["format"] == "csv":
        if "csv" not in top or "jsonl" in top:
            fail("E_CONFIG", "CSV config must have csv and must not have jsonl")
        csv_cfg = _exact_keys(
            top["csv"],
            {"header", "dialect", "blank_row", "field_trim"},
            "$.csv",
        )
        header = csv_cfg["header"]
        if (
            not isinstance(header, list)
            or not header
            or any(not isinstance(item, str) or not item for item in header)
            or len(set(header)) != len(header)
        ):
            fail("E_CONFIG", "$.csv.header must contain unique non-empty strings")
        if (
            csv_cfg["dialect"] != "rfc4180"
            or csv_cfg["blank_row"] != "reject"
            or csv_cfg["field_trim"] != "none"
        ):
            fail("E_CONFIG", "$.csv policy is unsupported")
        if origin["line_count"] != origin["record_count"] + 1:
            fail("E_CONFIG", "CSV line_count must equal record_count + header")
        _validate_mapping_sources(
            mapping,
            set(header),
            config["timestamp"]["field"],
            None,
        )
    else:
        if "jsonl" not in top or "csv" in top:
            fail("E_CONFIG", "JSONL config must have jsonl and must not have csv")
        jsonl_cfg = _exact_keys(
            top["jsonl"],
            {"required_keys", "key_types", "duplicate_key", "blank_row"},
            "$.jsonl",
            {"nested_hash_ids"},
        )
        required_keys = jsonl_cfg["required_keys"]
        if (
            not isinstance(required_keys, list)
            or not required_keys
            or any(not isinstance(item, str) or not item for item in required_keys)
            or len(set(required_keys)) != len(required_keys)
        ):
            fail("E_CONFIG", "$.jsonl.required_keys must contain unique strings")
        key_types = jsonl_cfg["key_types"]
        if not isinstance(key_types, dict) or set(key_types) != set(required_keys):
            fail("E_CONFIG", "$.jsonl.key_types must exactly cover required_keys")
        if any(
            value not in ("string", "integer", "number", "boolean", "object", "array")
            for value in key_types.values()
        ):
            fail("E_CONFIG", "$.jsonl.key_types contains an unsupported type")
        if jsonl_cfg["duplicate_key"] != "reject":
            fail("E_CONFIG", "$.jsonl.duplicate_key must equal reject")
        blank_policy = jsonl_cfg["blank_row"]
        if blank_policy not in ("reject", "reject_except_single_terminal"):
            fail("E_CONFIG", "$.jsonl.blank_row is unsupported")
        if blank_policy == "reject_except_single_terminal":
            if source != "ragpulse":
                fail("E_CONFIG", "terminal blank exception is restricted to ragpulse")
            if origin["line_count"] != origin["record_count"] + 1:
                fail("E_CONFIG", "ragpulse terminal blank requires line_count=record_count+1")
        elif origin["line_count"] != origin["record_count"]:
            fail("E_CONFIG", "JSONL line_count must equal record_count")
        if "nested_hash_ids" not in jsonl_cfg:
            fail("E_CONFIG", "RAGPulse config requires nested_hash_ids")
        if "nested_hash_ids" in jsonl_cfg:
            nested = _exact_keys(
                jsonl_cfg["nested_hash_ids"],
                {"required_keys", "value_type", "empty_allowed"},
                "$.jsonl.nested_hash_ids",
            )
            nested_keys = nested["required_keys"]
            empty_allowed = nested["empty_allowed"]
            if (
                not isinstance(nested_keys, list)
                or not nested_keys
                or len(set(nested_keys)) != len(nested_keys)
                or any(not isinstance(item, str) or not item for item in nested_keys)
            ):
                fail("E_CONFIG", "$.jsonl.nested_hash_ids.required_keys is invalid")
            if nested["value_type"] != "list_of_int":
                fail("E_CONFIG", "$.jsonl.nested_hash_ids.value_type is unsupported")
            if (
                not isinstance(empty_allowed, list)
                or len(set(empty_allowed)) != len(empty_allowed)
                or any(item not in nested_keys for item in empty_allowed)
            ):
                fail("E_CONFIG", "$.jsonl.nested_hash_ids.empty_allowed is invalid")
        if mapping["source_fields"]:
            fail("E_CONFIG", "JSONL source_fields mappings are not implemented")
        _validate_mapping_sources(
            mapping,
            set(required_keys),
            config["timestamp"]["field"],
            set(jsonl_cfg["nested_hash_ids"]["required_keys"]),
        )

    return top


def _validate_value_mapping(spec: Any, path: str) -> None:
    if not isinstance(spec, dict):
        fail("E_CONFIG", f"{path} must be an object")
    if set(spec) == {"const"}:
        _require_string(spec["const"], f"{path}.const")
        return
    _exact_keys(spec, {"from", "value_map"}, path)
    _require_string(spec["from"], f"{path}.from")
    if (
        not isinstance(spec["value_map"], dict)
        or not spec["value_map"]
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not value
            for key, value in spec["value_map"].items()
        )
    ):
        fail("E_CONFIG", f"{path}.value_map is invalid")


def _validate_retrieved_chunks(spec: Any) -> None:
    if not isinstance(spec, dict):
        fail("E_CONFIG", "$.mapping.retrieved_chunks must be an object")
    if set(spec) == {"const"}:
        _require_int(spec["const"], "$.mapping.retrieved_chunks.const")
        return
    _exact_keys(spec, {"derive", "from"}, "$.mapping.retrieved_chunks")
    if spec["derive"] != "len" or not isinstance(spec["from"], str):
        fail("E_CONFIG", "$.mapping.retrieved_chunks derivation is unsupported")


def _validate_cache_keys(spec: Any) -> None:
    if not isinstance(spec, dict):
        fail("E_CONFIG", "$.mapping.cache_keys must be an object")
    if set(spec) == {"const_empty"}:
        if spec["const_empty"] is not True:
            fail("E_CONFIG", "$.mapping.cache_keys.const_empty must be true")
        return
    _exact_keys(spec, {"derive", "order", "separator"}, "$.mapping.cache_keys")
    if spec["derive"] != "namespaced":
        fail("E_CONFIG", "$.mapping.cache_keys derivation is unsupported")
    if (
        not isinstance(spec["order"], list)
        or not spec["order"]
        or len(set(spec["order"])) != len(spec["order"])
        or any(not isinstance(item, str) or not item for item in spec["order"])
    ):
        fail("E_CONFIG", "$.mapping.cache_keys.order is invalid")
    if not isinstance(spec["separator"], str):
        fail("E_CONFIG", "$.mapping.cache_keys.separator must be a string")


def _validate_source_fields(specs: Any) -> None:
    if not isinstance(specs, dict):
        fail("E_CONFIG", "$.mapping.source_fields must be an object")
    for name, spec in specs.items():
        _require_string(name, "$.mapping.source_fields key")
        item = _exact_keys(
            spec,
            {"from", "type", "transform"},
            f"$.mapping.source_fields.{name}",
        )
        _require_string(item["from"], f"$.mapping.source_fields.{name}.from")
        if item["type"] not in ("string", "integer", "boolean"):
            fail("E_CONFIG", f"$.mapping.source_fields.{name}.type is unsupported")
        if item["transform"] not in ("raw", "is_zero"):
            fail("E_CONFIG", f"$.mapping.source_fields.{name}.transform is unsupported")
        if item["transform"] == "is_zero" and item["type"] != "boolean":
            fail("E_CONFIG", f"$.mapping.source_fields.{name}.is_zero must be boolean")


def _validate_mapping_sources(
    mapping: dict[str, Any],
    fields: set[str],
    timestamp_field: str,
    nested_hash_fields: set[str] | None,
) -> None:
    direct_sources = [
        timestamp_field,
        mapping["input_tokens"]["from"],
        mapping["output_tokens"]["from"],
        mapping["session_id"]["from"],
    ]
    for name in ("service", "model_class"):
        if "from" in mapping[name]:
            direct_sources.append(mapping[name]["from"])
    direct_sources.extend(spec["from"] for spec in mapping["source_fields"].values())
    missing = sorted(set(direct_sources) - fields)
    if missing:
        fail("E_CONFIG", f"mapping references absent source fields: {missing}")

    retrieved = mapping["retrieved_chunks"]
    cache_keys = mapping["cache_keys"]
    if nested_hash_fields is None:
        if "const" not in retrieved or "const_empty" not in cache_keys:
            fail("E_CONFIG", "CSV supports only constant retrieval and empty cache keys")
        return

    if "derive" not in retrieved or retrieved["derive"] != "len":
        fail("E_CONFIG", "RAGPulse retrieved_chunks must use len derivation")
    parts = retrieved["from"].split(".")
    if (
        len(parts) != 2
        or parts[0] != "hash_ids"
        or parts[1] not in nested_hash_fields
    ):
        fail("E_CONFIG", "retrieved_chunks path must name a hash_ids list")
    if "derive" not in cache_keys or cache_keys["derive"] != "namespaced":
        fail("E_CONFIG", "RAGPulse cache_keys must use namespaced derivation")
    if set(cache_keys["order"]) != nested_hash_fields:
        fail("E_CONFIG", "cache_keys.order must exactly cover hash_ids keys")


@dataclass(frozen=True)
class SourceInspection:
    byte_count: int
    sha256: str
    line_count: int
    terminal_blank_lines: int


@dataclass(frozen=True)
class ParsedEvent:
    source_row_id: int
    source_t_us: int
    input_tokens: int
    output_tokens: int
    session_id: str | None
    service: str
    model_class: str
    retrieved_chunks: int
    cache_keys: list[str]
    source_fields: dict[str, Any]


@dataclass(frozen=True)
class ScanResult:
    record_count: int
    nonmonotonic_pairs: int
    bin_counts: dict[int, int]
    bin_loads: dict[int, int]


@dataclass(frozen=True)
class ScenarioWindow:
    scenario: str
    bin_index: int
    metric_value: int
    quantile_rank: int | None


def inspect_source(path: Path, config: dict[str, Any]) -> SourceInspection:
    digest = hashlib.sha256()
    byte_count = 0
    line_count = 0
    blank_lines: list[int] = []
    first_body: bytes | None = None
    try:
        with path.open("rb") as source:
            for line_index, raw_line in enumerate(source):
                if line_index == 0 and raw_line.startswith(b"\xef\xbb\xbf"):
                    fail("E_BOM", f"{path}: UTF-8 BOM is forbidden")
                digest.update(raw_line)
                byte_count += len(raw_line)
                line_count += 1
                body = _strip_source_newline(raw_line, path, line_index)
                if line_index == 0:
                    first_body = body
                if body == b"":
                    blank_lines.append(line_index)
        if line_count == 0:
            fail("E_EMPTY_SOURCE", f"{path}: source file is empty")
    except NormalizeError:
        raise
    except OSError as exc:
        fail("E_IO", f"cannot inspect {path}: {exc}")

    actual_sha = "sha256:" + digest.hexdigest()
    origin = config["origin"]
    if byte_count != origin["bytes"]:
        fail("E_SOURCE_BYTES", f"{path}: bytes {byte_count} != pinned {origin['bytes']}")
    if actual_sha != origin["sha256"]:
        fail("E_SOURCE_HASH", f"{path}: sha256 {actual_sha} != pinned {origin['sha256']}")
    if line_count != origin["line_count"]:
        fail("E_SOURCE_LINES", f"{path}: lines {line_count} != pinned {origin['line_count']}")

    if config["format"] == "csv":
        if blank_lines:
            fail("E_BLANK_ROW", f"{path}: blank physical line {blank_lines[0] + 1}")
        expected_header = ",".join(config["csv"]["header"]).encode("utf-8")
        if first_body != expected_header:
            fail("E_HEADER", f"{path}: header bytes do not match the pinned header")
        terminal_blank_lines = 0
    else:
        policy = config["jsonl"]["blank_row"]
        if policy == "reject":
            if blank_lines:
                fail("E_BLANK_ROW", f"{path}: blank physical line {blank_lines[0] + 1}")
            terminal_blank_lines = 0
        else:
            expected = [line_count - 1]
            if blank_lines != expected:
                fail(
                    "E_BLANK_ROW",
                    f"{path}: expected exactly one terminal blank line, got {blank_lines}",
                )
            terminal_blank_lines = 1

    return SourceInspection(byte_count, actual_sha, line_count, terminal_blank_lines)


def copy_source_snapshot(source_path: Path, snapshot_path: Path) -> None:
    try:
        with source_path.open("rb") as source, snapshot_path.open("xb") as snapshot:
            shutil.copyfileobj(source, snapshot, length=1024 * 1024)
            snapshot.flush()
            os.fsync(snapshot.fileno())
    except OSError as exc:
        fail("E_IO", f"cannot snapshot {source_path}: {exc}")


def _strip_source_newline(raw_line: bytes, path: Path, line_index: int) -> bytes:
    if raw_line.endswith(b"\r\n"):
        body = raw_line[:-2]
    elif raw_line.endswith(b"\n"):
        body = raw_line[:-1]
    else:
        body = raw_line
    if b"\r" in body:
        fail("E_NEWLINE", f"{path}: lone CR on physical line {line_index + 1}")
    return body


def parse_timestamp(value: Any, config: dict[str, Any], row_id: int) -> int:
    if not isinstance(value, str):
        fail("E_TIMESTAMP", f"source_row_id={row_id}: timestamp must be a string")
    timestamp = config["timestamp"]
    if re.fullmatch(timestamp["grammar"], value) is None:
        fail("E_TIMESTAMP", f"source_row_id={row_id}: malformed timestamp {value!r}")
    try:
        result = int(
            (Decimal(value) * Decimal(1_000_000)).quantize(
                Decimal(1),
                rounding=ROUND_HALF_UP,
            )
        )
    except (InvalidOperation, ValueError) as exc:
        fail("E_TIMESTAMP", f"source_row_id={row_id}: invalid timestamp: {exc}")
    if not 0 <= result <= timestamp["overflow_max"]:
        fail("E_TIMESTAMP_OVERFLOW", f"source_row_id={row_id}: timestamp overflow")
    return result


def parse_integer_text(
    value: Any,
    grammar: str,
    row_id: int,
    field: str,
    maximum: int = SAFE_MAX,
) -> int:
    if not isinstance(value, str) or re.fullmatch(grammar, value) is None:
        fail("E_INTEGER", f"source_row_id={row_id}: malformed {field}={value!r}")
    result = int(value)
    if not 0 <= result <= maximum:
        fail("E_INTEGER", f"source_row_id={row_id}: {field} is outside [0,{maximum}]")
    return result


def _mapped_string(row: dict[str, str], spec: dict[str, Any], row_id: int, name: str) -> str:
    if "const" in spec:
        return spec["const"]
    value = row[spec["from"]]
    try:
        return spec["value_map"][value]
    except KeyError:
        fail("E_VALUE_MAP", f"source_row_id={row_id}: unmapped {name} value {value!r}")


def _source_fields_csv(
    row: dict[str, str],
    specs: dict[str, Any],
    row_id: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, spec in specs.items():
        value = row[spec["from"]]
        if spec["transform"] == "is_zero":
            parsed = parse_integer_text(value, r"^[0-9]+$", row_id, spec["from"])
            result[name] = parsed == 0
        elif spec["type"] == "integer":
            result[name] = parse_integer_text(value, r"^[0-9]+$", row_id, spec["from"])
        elif spec["type"] == "string":
            result[name] = value
        else:
            fail("E_CONFIG", f"unsupported source field mapping for {name}")
    return result


def iter_csv_events(path: Path, config: dict[str, Any]) -> Iterator[ParsedEvent]:
    header = config["csv"]["header"]
    mapping = config["mapping"]
    try:
        source = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        fail("E_IO", f"cannot open {path}: {exc}")
    with source:
        reader = csv.reader(
            source,
            delimiter=",",
            quotechar='"',
            doublequote=True,
            strict=True,
        )
        try:
            parsed_header = next(reader)
        except (StopIteration, csv.Error) as exc:
            fail("E_CSV", f"{path}: cannot read CSV header: {exc}")
        if parsed_header != header:
            fail("E_HEADER", f"{path}: parsed CSV header does not match config")
        try:
            for row_id, values in enumerate(reader):
                if len(values) != len(header):
                    fail(
                        "E_CSV",
                        f"source_row_id={row_id}: expected {len(header)} fields, got {len(values)}",
                    )
                row = dict(zip(header, values))
                input_spec = mapping["input_tokens"]
                output_spec = mapping["output_tokens"]
                input_tokens = parse_integer_text(
                    row[input_spec["from"]],
                    input_spec.get("grammar", r"^[0-9]+$"),
                    row_id,
                    input_spec["from"],
                    100_000_000,
                )
                output_tokens = parse_integer_text(
                    row[output_spec["from"]],
                    output_spec.get("grammar", r"^[0-9]+$"),
                    row_id,
                    output_spec["from"],
                    100_000_000,
                )
                session_value = row[mapping["session_id"]["from"]]
                session_id = (
                    None
                    if mapping["session_id"]["blank_to_null"] and session_value == ""
                    else session_value
                )
                if session_id is not None and len(session_id) > 512:
                    fail("E_RECORD", f"source_row_id={row_id}: session_id too long")
                yield ParsedEvent(
                    source_row_id=row_id,
                    source_t_us=parse_timestamp(
                        row[config["timestamp"]["field"]],
                        config,
                        row_id,
                    ),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    session_id=session_id,
                    service=_mapped_string(row, mapping["service"], row_id, "service"),
                    model_class=_mapped_string(
                        row,
                        mapping["model_class"],
                        row_id,
                        "model_class",
                    ),
                    retrieved_chunks=mapping["retrieved_chunks"]["const"],
                    cache_keys=[],
                    source_fields=_source_fields_csv(
                        row,
                        mapping["source_fields"],
                        row_id,
                    ),
                )
        except NormalizeError:
            raise
        except (csv.Error, UnicodeDecodeError) as exc:
            fail("E_CSV", f"{path}: malformed CSV: {exc}")


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "integer":
        return _is_int(value)
    if expected == "number":
        return _is_int(value) or isinstance(value, float)
    if expected == "boolean":
        return type(value) is bool
    if expected == "string":
        return isinstance(value, str)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return False


def iter_jsonl_events(path: Path, config: dict[str, Any]) -> Iterator[ParsedEvent]:
    jsonl_cfg = config["jsonl"]
    required_keys = jsonl_cfg["required_keys"]
    expected_keys = set(required_keys)
    mapping = config["mapping"]
    row_id = 0
    try:
        source: BinaryIO = path.open("rb")
    except OSError as exc:
        fail("E_IO", f"cannot open {path}: {exc}")
    with source:
        for physical_index, raw_line in enumerate(source):
            body = _strip_source_newline(raw_line, path, physical_index)
            if body == b"":
                continue
            obj = load_json_strict_bytes(body, f"{path}:{physical_index + 1}")
            if not isinstance(obj, dict) or set(obj) != expected_keys:
                fail(
                    "E_JSON_KEYS",
                    f"source_row_id={row_id}: keys do not exactly match required_keys",
                )
            for key, expected_type in jsonl_cfg["key_types"].items():
                if not _json_type_matches(obj[key], expected_type):
                    fail(
                        "E_JSON_TYPE",
                        f"source_row_id={row_id}: {key} must be {expected_type}",
                    )
            nested_cfg = jsonl_cfg.get("nested_hash_ids")
            hash_ids = obj["hash_ids"]
            if nested_cfg is not None:
                nested_keys = nested_cfg["required_keys"]
                if not isinstance(hash_ids, dict) or set(hash_ids) != set(nested_keys):
                    fail(
                        "E_JSON_KEYS",
                        f"source_row_id={row_id}: hash_ids keys mismatch",
                    )
                for key in nested_keys:
                    values = hash_ids[key]
                    if (
                        not isinstance(values, list)
                        or any(not _is_int(item) for item in values)
                    ):
                        fail(
                            "E_JSON_TYPE",
                            f"source_row_id={row_id}: hash_ids.{key} must be list_of_int",
                        )
                    if not values and key not in nested_cfg["empty_allowed"]:
                        fail(
                            "E_JSON_TYPE",
                            f"source_row_id={row_id}: hash_ids.{key} must be non-empty",
                        )

            input_value = obj[mapping["input_tokens"]["from"]]
            output_value = obj[mapping["output_tokens"]["from"]]
            if not _is_int(input_value) or not 0 <= input_value <= 100_000_000:
                fail("E_INTEGER", f"source_row_id={row_id}: invalid input_tokens")
            if not _is_int(output_value) or not 0 <= output_value <= 100_000_000:
                fail("E_INTEGER", f"source_row_id={row_id}: invalid output_tokens")
            timestamp_value = obj[config["timestamp"]["field"]]
            session_value = obj[mapping["session_id"]["from"]]
            session_id = (
                None
                if mapping["session_id"]["blank_to_null"] and session_value == ""
                else session_value
            )
            if session_id is not None and len(session_id) > 512:
                fail("E_RECORD", f"source_row_id={row_id}: session_id too long")

            retrieved_spec = mapping["retrieved_chunks"]
            if "const" in retrieved_spec:
                retrieved_chunks = retrieved_spec["const"]
            else:
                parent, child = retrieved_spec["from"].split(".", 1)
                if parent != "hash_ids" or child not in hash_ids:
                    fail("E_CONFIG", "unsupported retrieved_chunks path")
                retrieved_chunks = len(hash_ids[child])
            if retrieved_chunks > 1_000_000:
                fail("E_RECORD", f"source_row_id={row_id}: too many retrieved chunks")

            cache_spec = mapping["cache_keys"]
            if "const_empty" in cache_spec:
                cache_keys: list[str] = []
            else:
                cache_keys = []
                separator = cache_spec["separator"]
                for key in cache_spec["order"]:
                    if key not in hash_ids:
                        fail("E_CONFIG", f"cache-key source {key} is absent")
                    for value in hash_ids[key]:
                        cache_key = f"{key}{separator}{value}"
                        if len(cache_key) > 256:
                            fail("E_RECORD", f"source_row_id={row_id}: cache key too long")
                        cache_keys.append(cache_key)
                if len(cache_keys) > 4096:
                    fail("E_RECORD", f"source_row_id={row_id}: too many cache keys")

            yield ParsedEvent(
                source_row_id=row_id,
                source_t_us=parse_timestamp(timestamp_value, config, row_id),
                input_tokens=input_value,
                output_tokens=output_value,
                session_id=session_id,
                service=_mapped_json_string(obj, mapping["service"], row_id, "service"),
                model_class=_mapped_json_string(
                    obj,
                    mapping["model_class"],
                    row_id,
                    "model_class",
                ),
                retrieved_chunks=retrieved_chunks,
                cache_keys=cache_keys,
                source_fields={},
            )
            row_id += 1


def _mapped_json_string(
    row: dict[str, Any],
    spec: dict[str, Any],
    row_id: int,
    name: str,
) -> str:
    if "const" in spec:
        return spec["const"]
    value = row[spec["from"]]
    if not isinstance(value, str):
        fail("E_JSON_TYPE", f"source_row_id={row_id}: {spec['from']} must be string")
    try:
        return spec["value_map"][value]
    except KeyError:
        fail("E_VALUE_MAP", f"source_row_id={row_id}: unmapped {name} value {value!r}")


def iter_events(path: Path, config: dict[str, Any]) -> Iterator[ParsedEvent]:
    if config["format"] == "csv":
        yield from iter_csv_events(path, config)
    else:
        yield from iter_jsonl_events(path, config)


def scan_events(path: Path, config: dict[str, Any]) -> ScanResult:
    bin_counts: dict[int, int] = {}
    bin_loads: dict[int, int] = {}
    count = 0
    previous_t: int | None = None
    nonmonotonic_pairs = 0
    for event in iter_events(path, config):
        if event.source_row_id != count:
            fail("E_ROW_ID", f"expected source_row_id={count}, got {event.source_row_id}")
        if previous_t is not None and event.source_t_us < previous_t:
            nonmonotonic_pairs += 1
            if config["timestamp"]["policy"] == "require_nondecreasing":
                fail(
                    "E_TIMESTAMP_ORDER",
                    f"source_row_id={event.source_row_id}: timestamp decreased",
                )
        previous_t = event.source_t_us
        bin_index = event.source_t_us // WINDOW_US
        load = event.input_tokens + event.output_tokens
        next_load = bin_loads.get(bin_index, 0) + load
        if next_load > SAFE_MAX:
            fail("E_LOAD_OVERFLOW", f"bin {bin_index}: offered-token load overflow")
        bin_loads[bin_index] = next_load
        bin_counts[bin_index] = bin_counts.get(bin_index, 0) + 1
        count += 1
    if count != config["origin"]["record_count"]:
        fail(
            "E_SOURCE_RECORDS",
            f"parsed records {count} != pinned {config['origin']['record_count']}",
        )
    if not bin_counts:
        fail("E_EMPTY_SOURCE", "source has no records")
    return ScanResult(count, nonmonotonic_pairs, bin_counts, bin_loads)


def select_windows(scan: ScanResult) -> dict[str, ScenarioWindow]:
    bins = sorted(scan.bin_loads, key=lambda index: (scan.bin_loads[index], index))
    count = len(bins)
    selected: dict[str, ScenarioWindow] = {}
    for scenario, (numerator, denominator) in QUANTILES.items():
        rank = (numerator * count + denominator - 1) // denominator
        rank = min(max(rank, 1), count)
        bin_index = bins[rank - 1]
        selected[scenario] = ScenarioWindow(
            scenario,
            bin_index,
            scan.bin_loads[bin_index],
            rank,
        )
    burst_bin = max(scan.bin_loads, key=lambda index: (scan.bin_loads[index], -index))
    selected["burst"] = ScenarioWindow(
        "burst",
        burst_bin,
        scan.bin_loads[burst_bin],
        None,
    )
    return selected


def make_record(
    event: ParsedEvent,
    config: dict[str, Any],
    bin_index: int,
) -> dict[str, Any]:
    start_us = bin_index * WINDOW_US
    rebased = event.source_t_us - start_us
    if not 0 <= rebased < WINDOW_US:
        fail("E_WINDOW", f"source_row_id={event.source_row_id}: outside selected bin")
    record = {
        "schema_version": 1,
        "event_id": f"{config['source']}:{event.source_row_id}",
        "source": config["source"],
        "provenance": config["provenance"],
        "t_us": rebased,
        "service": event.service,
        "model_class": event.model_class,
        "session_id": event.session_id,
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "images": config["gate_a"]["images"],
        "audio_ms": config["gate_a"]["audio_ms"],
        "retrieved_chunks": event.retrieved_chunks,
        "cache_keys": event.cache_keys,
        "observed_latency_us": config["gate_a"]["observed_latency_us"],
        "priority_class": config["gate_a"]["priority_class"],
        "deadline_us": config["gate_a"]["deadline_us"],
        "priority_provenance": config["gate_a"]["priority_provenance"],
        "deadline_provenance": config["gate_a"]["deadline_provenance"],
        "source_fields": event.source_fields,
    }
    validate_record(record)
    return record


def validate_record(record: dict[str, Any]) -> None:
    if len(record["event_id"]) > 512 or len(record["source"]) > 128:
        fail("E_RECORD", f"{record['event_id']}: identifier exceeds schema bound")
    for key in ("service", "model_class"):
        if not isinstance(record[key], str) or not 1 <= len(record[key]) <= 128:
            fail("E_RECORD", f"{record['event_id']}: invalid {key}")
    for key, maximum in (
        ("t_us", SAFE_MAX),
        ("input_tokens", 100_000_000),
        ("output_tokens", 100_000_000),
        ("images", 1_000_000),
        ("audio_ms", 4_294_967_295),
        ("retrieved_chunks", 1_000_000),
    ):
        value = record[key]
        if not _is_int(value) or not 0 <= value <= maximum:
            fail("E_RECORD", f"{record['event_id']}: invalid {key}")
    if record["session_id"] is not None and (
        not isinstance(record["session_id"], str) or len(record["session_id"]) > 512
    ):
        fail("E_RECORD", f"{record['event_id']}: invalid session_id")
    if not isinstance(record["cache_keys"], list) or len(record["cache_keys"]) > 4096:
        fail("E_RECORD", f"{record['event_id']}: invalid cache_keys")
    if any(not isinstance(key, str) or len(key) > 256 for key in record["cache_keys"]):
        fail("E_RECORD", f"{record['event_id']}: invalid cache key")
    if (
        record["deadline_us"] is not None
        or record["deadline_provenance"] != "none"
        or record["priority_class"] is not None
        or record["priority_provenance"] != "none"
    ):
        fail("E_RECORD", f"{record['event_id']}: Gate-A SLO fields are not frozen")
    source_fields = record["source_fields"]
    if not isinstance(source_fields, dict):
        fail("E_RECORD", f"{record['event_id']}: source_fields must be an object")
    for key, value in source_fields.items():
        if not isinstance(key, str) or not (
            value is None
            or isinstance(value, (str, bool))
            or _is_int(value)
        ):
            fail("E_RECORD", f"{record['event_id']}: invalid source_fields.{key}")
    _reject_unsupported_json(record, "$")


def collect_records(
    path: Path,
    config: dict[str, Any],
    windows: dict[str, ScenarioWindow],
    scenarios: list[str],
) -> dict[str, list[dict[str, Any]]]:
    by_bin: dict[int, list[ParsedEvent]] = {
        windows[scenario].bin_index: [] for scenario in scenarios
    }
    count = 0
    for event in iter_events(path, config):
        count += 1
        bin_index = event.source_t_us // WINDOW_US
        if bin_index in by_bin:
            by_bin[bin_index].append(event)
    if count != config["origin"]["record_count"]:
        fail("E_SOURCE_RECORDS", f"second pass parsed {count} records")

    result: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        window = windows[scenario]
        events = by_bin[window.bin_index]
        if config["timestamp"]["policy"] == "sort_stable":
            events.sort(key=lambda event: (event.source_t_us, event.source_row_id))
        records = [make_record(event, config, window.bin_index) for event in events]
        records.sort(
            key=lambda record: (
                record["t_us"],
                int(record["event_id"].rsplit(":", 1)[1]),
            )
        )
        if not records:
            fail("E_EMPTY_OUTPUT", f"scenario {scenario} selected an empty bin")
        event_ids = [record["event_id"] for record in records]
        if len(set(event_ids)) != len(event_ids):
            fail("E_DUPLICATE_EVENT_ID", f"scenario {scenario} has duplicate event IDs")
        result[scenario] = records
    return result


def build_manifest(
    inspection: SourceInspection,
    config: dict[str, Any],
    scan: ScanResult,
    window: ScenarioWindow,
    records: list[dict[str, Any]],
    output_sha256: str,
    code_version: str,
    config_hash: str,
) -> dict[str, Any]:
    row_ids = [int(record["event_id"].rsplit(":", 1)[1]) for record in records]
    start_us = window.bin_index * WINDOW_US
    end_us = (window.bin_index + 1) * WINDOW_US
    if end_us > SAFE_MAX:
        fail("E_WINDOW_OVERFLOW", f"scenario {window.scenario}: window end overflow")
    window_record: dict[str, Any] = {
        "rule": "aligned_15m_offered_tokens_nearest_rank_v1",
        "load_metric": "offered_tokens",
        "scenario": window.scenario,
        "bin_index": window.bin_index,
        "t_start_us": start_us,
        "t_end_us": end_us,
        "metric_value": window.metric_value,
        "n_nonempty_bins": len(scan.bin_counts),
        "source_row_first": min(row_ids),
        "source_row_last": max(row_ids),
        "timestamp_policy": config["timestamp"]["policy"],
        "input_nonmonotonic_pairs": scan.nonmonotonic_pairs,
    }
    if window.quantile_rank is not None:
        window_record["quantile_rank"] = window.quantile_rank
    return {
        "schema_version": 1,
        "provenance": config["provenance"],
        "normalizer_version": code_version,
        "normalization_config_hash": config_hash,
        "output_row_count": len(records),
        "output_sha256": output_sha256,
        "source_url": config["origin"]["url"],
        "source_revision": config["origin"]["revision"],
        "license": config["origin"]["license"],
        "source_file": config["origin"]["filename"],
        "source_bytes": inspection.byte_count,
        "source_sha256": inspection.sha256,
        "window": window_record,
        "filters": [],
        "exclusion_counts": {},
        "time_scale_num": 1,
        "time_scale_den": 1,
    }


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_fsync(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        fail("E_IO", f"cannot write staged file {path}: {exc}")


def stage_scenario(
    staging_dir: Path,
    final_dir: Path,
    scenario: str,
    records: list[dict[str, Any]],
    inspection: SourceInspection,
    config: dict[str, Any],
    scan: ScanResult,
    window: ScenarioWindow,
    code_version: str,
    config_hash: str,
) -> dict[str, Any]:
    stem = f"{config['source']}.{scenario}"
    output_name = f"{stem}.jsonl"
    manifest_name = f"{stem}.manifest.json"
    output_bytes = b"".join(canonical_json(record) + b"\n" for record in records)
    output_sha = sha256_bytes(output_bytes)
    manifest = build_manifest(
        inspection,
        config,
        scan,
        window,
        records,
        output_sha,
        code_version,
        config_hash,
    )
    manifest_bytes = canonical_json(manifest)
    manifest_sha = sha256_bytes(manifest_bytes)
    _write_bytes_fsync(staging_dir / output_name, output_bytes)
    _write_bytes_fsync(staging_dir / manifest_name, manifest_bytes + b"\n")
    return {
        "scenario": scenario,
        "output_path": str(final_dir / output_name),
        "manifest_path": str(final_dir / manifest_name),
        "artifact_output_path": output_name,
        "artifact_manifest_path": manifest_name,
        "output_row_count": len(records),
        "output_sha256": output_sha,
        "sidecar_manifest_sha256": manifest_sha,
        "bin_index": window.bin_index,
        "metric_value": window.metric_value,
    }


def build_artifact_manifest(
    config: dict[str, Any],
    config_path: Path,
    scenarios: list[str],
    outputs: list[dict[str, Any]],
    code_version: str,
    config_hash: str,
) -> dict[str, Any]:
    artifact_outputs = [
        {
            "path": output["artifact_output_path"],
            "sha256": output["output_sha256"],
            "sidecar_manifest_sha256": output["sidecar_manifest_sha256"],
        }
        for output in sorted(outputs, key=lambda item: item["artifact_output_path"])
    ]
    replay_preimage = "\n".join(
        output["sha256"] for output in artifact_outputs
    ).encode("ascii")
    run_preimage = canonical_json(
        {
            "source_sha256": config["origin"]["sha256"],
            "config_hash": config_hash,
            "code_version": code_version,
            "scenarios": scenarios,
        }
    )
    return {
        "schema_version": 1,
        "run_id": "normalize-" + hashlib.sha256(run_preimage).hexdigest()[:24],
        "kind": "normalize",
        "inputs": [
            {
                "role": "source",
                "path": config["origin"]["filename"],
                "sha256": config["origin"]["sha256"],
            },
            {
                "role": "normalization_config",
                "path": f"configs/{config_path.name}",
                "sha256": config_hash,
            },
        ],
        "code_version": code_version,
        "config_hash": config_hash,
        "seed": None,
        "outputs": artifact_outputs,
        "deterministic_replay_sha256": sha256_bytes(replay_preimage),
        "gate_results": {
            "atomic_run_publish": True,
            "direct_source_execution": DIRECT_SOURCE_EXECUTION,
            "source_snapshot_verified": True,
        },
    }


def publish_run(
    output_dir: Path,
    scenarios: list[str],
    records: dict[str, list[dict[str, Any]]],
    inspection: SourceInspection,
    config: dict[str, Any],
    config_path: Path,
    scan: ScanResult,
    windows: dict[str, ScenarioWindow],
    code_version: str,
    config_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if output_dir.exists():
        fail("E_EXISTS", f"output directory already exists: {output_dir}")
    parent = output_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".tmp", dir=parent)
        )
    except OSError as exc:
        fail("E_IO", f"cannot create staging directory for {output_dir}: {exc}")

    published = False
    try:
        outputs = [
            stage_scenario(
                staging_dir,
                output_dir,
                scenario,
                records[scenario],
                inspection,
                config,
                scan,
                windows[scenario],
                code_version,
                config_hash,
            )
            for scenario in scenarios
        ]
        artifact = build_artifact_manifest(
            config,
            config_path,
            scenarios,
            outputs,
            code_version,
            config_hash,
        )
        artifact_bytes = canonical_json(artifact)
        _write_bytes_fsync(
            staging_dir / "normalize.artifact.json",
            artifact_bytes + b"\n",
        )
        _fsync_directory(staging_dir)
        if normalizer_version() != code_version:
            fail("E_CODE_CHANGED", "normalizer source changed during the run")
        if output_dir.exists():
            fail("E_EXISTS", f"output directory appeared during run: {output_dir}")
        os.rename(staging_dir, output_dir)
        published = True
        _fsync_directory(parent)
        for output in outputs:
            output.pop("artifact_output_path")
            output.pop("artifact_manifest_path")
        return outputs, {
            "artifact_path": str(output_dir / "normalize.artifact.json"),
            "artifact_manifest_sha256": sha256_bytes(artifact_bytes),
            "deterministic_replay_sha256": artifact["deterministic_replay_sha256"],
        }
    except NormalizeError:
        raise
    except OSError as exc:
        fail("E_IO", f"cannot publish run {output_dir}: {exc}")
    finally:
        if not published:
            shutil.rmtree(staging_dir, ignore_errors=True)


def scan_summary(
    inspection: SourceInspection,
    config: dict[str, Any],
    scan: ScanResult,
    windows: dict[str, ScenarioWindow],
) -> dict[str, Any]:
    return {
        "source": config["source"],
        "source_bytes": inspection.byte_count,
        "source_sha256": inspection.sha256,
        "line_count": inspection.line_count,
        "record_count": scan.record_count,
        "terminal_blank_lines": inspection.terminal_blank_lines,
        "timestamp_policy": config["timestamp"]["policy"],
        "input_nonmonotonic_pairs": scan.nonmonotonic_pairs,
        "n_nonempty_bins": len(scan.bin_counts),
        "scenarios": {
            scenario: {
                "bin_index": window.bin_index,
                "request_count": scan.bin_counts[window.bin_index],
                "metric_value": window.metric_value,
                "quantile_rank": window.quantile_rank,
            }
            for scenario, window in windows.items()
        },
    }


def _normalize_impl(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_file).resolve()
    config_path = Path(args.config).resolve()
    code_version = STARTUP_NORMALIZER_VERSION
    config = validate_config(load_json_strict(config_path))
    config_hash = sha256_bytes(canonical_json(config))
    with tempfile.TemporaryDirectory(prefix="s8_normalize_snapshot_") as temporary:
        snapshot_path = Path(temporary) / "source.snapshot"
        copy_source_snapshot(source_path, snapshot_path)
        inspection = inspect_source(snapshot_path, config)
        scan = scan_events(snapshot_path, config)
        windows = select_windows(scan)
        summary = scan_summary(inspection, config, scan, windows)
        if args.scan_only:
            if normalizer_version() != code_version:
                fail("E_CODE_CHANGED", "normalizer source changed during the scan")
            return {"status": "scan_only", **summary}

        scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        records = collect_records(snapshot_path, config, windows, scenarios)
        output_dir = Path(args.output_dir).resolve()
        outputs, artifact_info = publish_run(
            output_dir,
            scenarios,
            records,
            inspection,
            config,
            config_path,
            scan,
            windows,
            code_version,
            config_hash,
        )
    return {
        "status": "normalized",
        **summary,
        "outputs": outputs,
        **artifact_info,
    }


def normalize(args: argparse.Namespace) -> dict[str, Any]:
    if not DIRECT_SOURCE_EXECUTION:
        fail(
            "E_EXECUTION_PROVENANCE",
            "certified normalization requires direct CLI source execution",
        )
    return _normalize_impl(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--scan-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.scan_only and not args.output_dir:
        parser.error("--output-dir is required unless --scan-only is used")
    try:
        result = normalize(args)
    except NormalizeError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"E_INTERNAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(canonical_json(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
