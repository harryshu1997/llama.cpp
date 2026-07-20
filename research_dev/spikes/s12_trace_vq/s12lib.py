#!/usr/bin/env python3
"""Shared fail-closed helpers for the S12 trace-driven VQ replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


MAX_SAFE_INT = 9007199254740991
MIB = 1024 * 1024


class S12Error(ValueError):
    pass


def is_int(value: Any) -> bool:
    return type(value) is int


def require_int(name: str, value: Any, minimum: int = 0, maximum: int = MAX_SAFE_INT) -> int:
    if not is_int(value) or value < minimum or value > maximum:
        raise S12Error(f"{name}: expected integer in [{minimum}, {maximum}]")
    return value


def require_str(name: str, value: Any, allowed: Iterable[str] | None = None) -> str:
    if type(value) is not str or not value:
        raise S12Error(f"{name}: expected non-empty string")
    if allowed is not None and value not in set(allowed):
        raise S12Error(f"{name}: unsupported value {value!r}")
    return value


def require_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise S12Error(f"{name}: expected boolean")
    return value


def require_exact_keys(name: str, obj: Any, required: set[str]) -> dict[str, Any]:
    if type(obj) is not dict:
        raise S12Error(f"{name}: expected object")
    actual = set(obj)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise S12Error(f"{name}: key mismatch missing={missing} extra={extra}")
    return obj


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise S12Error(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def strict_json_loads(text: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                S12Error(f"non-finite JSON constant: {value}")
            ),
        )
    except S12Error:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise S12Error(f"invalid JSON: {exc}") from exc


def load_json(path: str | Path) -> Any:
    p = Path(path)
    try:
        return strict_json_loads(p.read_text(encoding="ascii"))
    except (OSError, UnicodeError) as exc:
        raise S12Error(f"{p}: cannot read ASCII JSON: {exc}") from exc


def read_bytes_once(path: str | Path) -> bytes:
    p = Path(path)
    try:
        return p.read_bytes()
    except OSError as exc:
        raise S12Error(f"{p}: cannot read bytes: {exc}") from exc


def parse_jsonl_bytes(data: bytes, source: str = "<snapshot>") -> list[Any]:
    try:
        lines = data.decode("ascii").splitlines()
    except UnicodeError as exc:
        raise S12Error(f"{source}: cannot decode ASCII JSONL: {exc}") from exc
    if not lines:
        raise S12Error(f"{source}: empty JSONL")
    records = []
    for line_no, line in enumerate(lines, 1):
        if not line:
            raise S12Error(f"{source}:{line_no}: empty JSONL record")
        try:
            records.append(strict_json_loads(line))
        except S12Error as exc:
            raise S12Error(f"{source}:{line_no}: {exc}") from exc
    return records


def read_jsonl_snapshot(path: str | Path) -> tuple[list[Any], str]:
    p = Path(path)
    data = read_bytes_once(p)
    return parse_jsonl_bytes(data, str(p)), sha256_bytes(data)


def load_jsonl(path: str | Path) -> list[Any]:
    records, _ = read_jsonl_snapshot(path)
    return records


def canonical_json(obj: Any) -> str:
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise S12Error(f"object is not canonical-JSON encodable: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    try:
        return sha256_bytes(p.read_bytes())
    except OSError as exc:
        raise S12Error(f"{p}: cannot hash: {exc}") from exc


def sha256_object(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("ascii"))


def write_canonical(path: str | Path, obj: Any) -> None:
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(canonical_json(obj) + "\n", encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise S12Error(f"{p}: cannot write canonical JSON: {exc}") from exc


def nearest_rank(values: list[int], numerator: int, denominator: int) -> int | None:
    if not values:
        return None
    require_int("quantile numerator", numerator, 1)
    require_int("quantile denominator", denominator, 1)
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    rank = max(1, min(len(ordered), rank))
    return ordered[rank - 1]


def checked_mul(name: str, *values: int) -> int:
    result = 1
    for value in values:
        require_int(name, value)
        result *= value
        if result > MAX_SAFE_INT:
            raise S12Error(f"{name}: integer overflow")
    return result


def _require_sha256(name: str, value: Any) -> str:
    value = require_str(name, value)
    if len(value) != 71 or not value.startswith("sha256:"):
        raise S12Error(f"{name}: expected sha256:<64 lowercase hex>")
    suffix = value[7:]
    if any(c not in "0123456789abcdef" for c in suffix):
        raise S12Error(f"{name}: expected sha256:<64 lowercase hex>")
    return value


def validate_profile(profile: Any, base_dir: str | Path | None = None, verify_artifacts: bool = False) -> dict[str, Any]:
    required = {
        "schema",
        "scope",
        "energy_status",
        "model",
        "route",
        "source_bindings",
        "rows",
    }
    obj = require_exact_keys("profile", profile, required)
    require_str("profile.schema", obj["schema"], {"s12-s11-batched-route-profile-v1"})
    require_str("profile.scope", obj["scope"], {"MECHANICS_ONLY"})
    require_str("profile.energy_status", obj["energy_status"], {"NOT_RUN"})

    model = require_exact_keys(
        "profile.model",
        obj["model"],
        {
            "activation_element_bytes",
            "activation_rows_per_request",
            "context_tokens",
            "embedding_width",
            "generated_tokens",
            "host_model_sha256",
            "model_id",
            "phone_weight_bytes",
            "prompt_sha256",
            "prompt_tokens",
        },
    )
    require_str("profile.model.model_id", model["model_id"])
    _require_sha256("profile.model.host_model_sha256", model["host_model_sha256"])
    _require_sha256("profile.model.prompt_sha256", model["prompt_sha256"])
    for key in (
        "activation_element_bytes",
        "activation_rows_per_request",
        "context_tokens",
        "embedding_width",
        "generated_tokens",
        "phone_weight_bytes",
        "prompt_tokens",
    ):
        require_int(f"profile.model.{key}", model[key], 1)
    if model["activation_rows_per_request"] != model["prompt_tokens"] + model["generated_tokens"] - 1:
        raise S12Error("profile.model.activation_rows_per_request: expected prompt_tokens + generated_tokens - 1")

    route = require_exact_keys(
        "profile.route",
        obj["route"],
        {"control", "phone", "phone_backend", "phone_device", "phone_layer_range"},
    )
    require_str("profile.route.control", route["control"], {"SERVER_ONLY"})
    require_str("profile.route.phone", route["phone"], {"A0_OP15"})
    require_str("profile.route.phone_backend", route["phone_backend"], {"HTP0"})
    require_str("profile.route.phone_device", route["phone_device"], {"OP15"})
    if (
        type(route["phone_layer_range"]) is not list
        or len(route["phone_layer_range"]) != 2
        or any(not is_int(v) for v in route["phone_layer_range"])
        or route["phone_layer_range"] != [0, 2]
    ):
        raise S12Error("profile.route.phone_layer_range: expected [0,2]")

    bindings = obj["source_bindings"]
    if type(bindings) is not list or not bindings:
        raise S12Error("profile.source_bindings: expected non-empty array")
    seen_binding_paths: set[str] = set()
    root = Path(base_dir) if base_dir is not None else None
    for index, binding in enumerate(bindings):
        entry = require_exact_keys(
            f"profile.source_bindings[{index}]",
            binding,
            {"kind", "path", "sha256"},
        )
        require_str(f"profile.source_bindings[{index}].kind", entry["kind"])
        path = require_str(f"profile.source_bindings[{index}].path", entry["path"])
        digest = _require_sha256(f"profile.source_bindings[{index}].sha256", entry["sha256"])
        if path in seen_binding_paths:
            raise S12Error(f"profile.source_bindings[{index}].path: duplicate")
        seen_binding_paths.add(path)
        if verify_artifacts:
            if root is None:
                raise S12Error("verify_artifacts requires base_dir")
            actual = sha256_file(root / path)
            if actual != digest:
                raise S12Error(f"profile.source_bindings[{index}]: digest mismatch")

    rows = obj["rows"]
    if type(rows) is not list or len(rows) != 5:
        raise S12Error("profile.rows: expected exactly five S11 rows")
    expected_batches = [1, 2, 4, 8, 16]
    for index, row in enumerate(rows):
        entry = require_exact_keys(
            f"profile.rows[{index}]",
            row,
            {
                "activation_bytes",
                "batch_size",
                "control_ready_hbm_mib",
                "energy_status",
                "exact_work",
                "latency_scope",
                "phone_route_group_us",
                "phone_route_ready_hbm_mib",
                "phone_stage_us",
                "plan_sha256",
                "plan_path",
                "server_group_us",
                "server_hbm_relief_mib",
                "server_tail_us",
                "summary_sha256",
                "summary_path",
            },
        )
        batch = require_int(f"profile.rows[{index}].batch_size", entry["batch_size"], 1)
        if batch != expected_batches[index]:
            raise S12Error("profile.rows: batches must be exactly [1,2,4,8,16]")
        for key in (
            "activation_bytes",
            "control_ready_hbm_mib",
            "phone_route_group_us",
            "phone_route_ready_hbm_mib",
            "phone_stage_us",
            "server_group_us",
            "server_hbm_relief_mib",
            "server_tail_us",
        ):
            require_int(f"profile.rows[{index}].{key}", entry[key], 1)
        expected_activation = checked_mul(
            "activation_bytes",
            batch,
            model["activation_rows_per_request"],
            model["embedding_width"],
            model["activation_element_bytes"],
        )
        if entry["activation_bytes"] != expected_activation:
            raise S12Error(f"profile.rows[{index}].activation_bytes: formula mismatch")
        if entry["control_ready_hbm_mib"] - entry["phone_route_ready_hbm_mib"] != entry["server_hbm_relief_mib"]:
            raise S12Error(f"profile.rows[{index}]: HBM relief mismatch")
        require_bool(f"profile.rows[{index}].exact_work", entry["exact_work"])
        if not entry["exact_work"]:
            raise S12Error(f"profile.rows[{index}].exact_work: expected true")
        require_str(f"profile.rows[{index}].energy_status", entry["energy_status"], {"NOT_RUN"})
        require_str(
            f"profile.rows[{index}].latency_scope",
            entry["latency_scope"],
            {"MECHANICS_ONLY_SINGLE_PROCESS_PAIR"},
        )
        for key in ("plan_path", "summary_path"):
            require_str(f"profile.rows[{index}].{key}", entry[key])
        for key in ("plan_sha256", "summary_sha256"):
            _require_sha256(f"profile.rows[{index}].{key}", entry[key])
        if verify_artifacts:
            if root is None:
                raise S12Error("verify_artifacts requires base_dir")
            if sha256_file(root / entry["plan_path"]) != entry["plan_sha256"]:
                raise S12Error(f"profile.rows[{index}].plan_sha256: digest mismatch")
            if sha256_file(root / entry["summary_path"]) != entry["summary_sha256"]:
                raise S12Error(f"profile.rows[{index}].summary_sha256: digest mismatch")
    return obj


def profile_rows_by_batch(profile: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {row["batch_size"]: row for row in profile["rows"]}
