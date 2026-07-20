#!/usr/bin/env python3
"""Canonical executable-input manifest for physical S15 launches."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass


SCHEMA = "s15-input-manifest-v1"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class InputManifestError(ValueError):
    pass


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("ascii")


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _strict_load(payload: bytes) -> dict:
    if type(payload) is not bytes or not payload or len(payload) > MAX_MANIFEST_BYTES:
        raise InputManifestError("manifest has an invalid byte length")

    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise InputManifestError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputManifestError(f"invalid manifest JSON: {exc}") from exc
    if type(value) is not dict or canonical(value) != payload:
        raise InputManifestError("manifest must be one canonical JSON object")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise InputManifestError(f"{name} must be a non-empty string")
    return value


def _sha256(name: str, value: object) -> str:
    text = _text(name, value)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise InputManifestError(f"{name} must be a lowercase sha256 digest")
    return text


@dataclass(frozen=True)
class InputManifest:
    sha256: str
    workload_id: str
    source_kind: str
    route_identity: dict
    request_ids: tuple[str, ...]
    inputs: tuple[bytes, ...]

    @property
    def repeated_input(self) -> bytes:
        if not self.inputs or any(item != self.inputs[0] for item in self.inputs):
            raise InputManifestError("current B32 route requires identical input bytes")
        return self.inputs[0]


def load_manifest(payload: bytes) -> InputManifest:
    value = _strict_load(payload)
    required = {
        "schema", "workload_id", "source_kind", "repeat_constraint",
        "route_identity", "requests",
    }
    if set(value) != required:
        raise InputManifestError("manifest has missing or unknown fields")
    if value["schema"] != SCHEMA:
        raise InputManifestError("unsupported manifest schema")
    workload_id = _text("workload_id", value["workload_id"])
    source_kind = _text("source_kind", value["source_kind"])
    if value["repeat_constraint"] != "all_input_bytes_identical":
        raise InputManifestError("unsupported repeat constraint")

    identity = value["route_identity"]
    identity_fields = {
        "route_id", "profile_id", "device_id", "model_id", "model_sha256",
        "shard_sha256", "layer_start", "layer_end",
    }
    if type(identity) is not dict or set(identity) != identity_fields:
        raise InputManifestError("route identity has missing or unknown fields")
    for field in ("route_id", "device_id", "model_id"):
        _text(field, identity[field])
    for field in ("profile_id", "model_sha256", "shard_sha256"):
        _sha256(field, identity[field])
    if type(identity["layer_start"]) is not int or type(identity["layer_end"]) is not int \
            or identity["layer_start"] < 0 or identity["layer_end"] <= identity["layer_start"]:
        raise InputManifestError("invalid route layer range")

    requests = value["requests"]
    if type(requests) is not list or not requests:
        raise InputManifestError("requests must be a non-empty list")
    request_ids = []
    inputs = []
    for request in requests:
        if type(request) is not dict or set(request) != {
            "request_id", "input_b64", "input_sha256"
        }:
            raise InputManifestError("request has missing or unknown fields")
        request_ids.append(_text("request_id", request["request_id"]))
        _sha256("input_sha256", request["input_sha256"])
        try:
            raw = base64.b64decode(request["input_b64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise InputManifestError("input_b64 is not canonical base64") from exc
        if not raw or base64.b64encode(raw).decode("ascii") != request["input_b64"]:
            raise InputManifestError("input bytes are empty or non-canonical")
        if digest(raw) != request["input_sha256"]:
            raise InputManifestError("input digest mismatch")
        inputs.append(raw)
    if len(set(request_ids)) != len(request_ids):
        raise InputManifestError("request IDs must be unique")

    result = InputManifest(
        digest(payload), workload_id, source_kind, dict(identity),
        tuple(request_ids), tuple(inputs),
    )
    result.repeated_input
    return result


def validate_launch_manifest(manifest: InputManifest, request: dict) -> bytes:
    if request.get("input_manifest_sha256") != manifest.sha256:
        raise InputManifestError("launch input-manifest digest mismatch")
    if request.get("request_ids") != list(manifest.request_ids):
        raise InputManifestError("launch request IDs do not match the input manifest")
    identity = manifest.route_identity
    for field in ("route_id", "profile_id", "device_id"):
        if request.get(field) != identity[field]:
            raise InputManifestError(f"launch {field} does not match the input manifest")
    if request.get("layer_range") != [identity["layer_start"], identity["layer_end"]]:
        raise InputManifestError("launch layer range does not match the input manifest")
    return manifest.repeated_input
