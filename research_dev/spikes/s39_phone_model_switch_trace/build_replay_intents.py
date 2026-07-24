#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


VERSION = "s39-active-trace-replay-v1"
HERE = Path(__file__).resolve().parent
DEFAULT_SELECTOR = HERE / "ACTIVE_TRACE.json"
DEFAULT_SHARD_MANIFEST = HERE / "SHARD_MANIFEST.json"

SHA256_RE = re.compile(r"[0-9a-f]{64}")
MODEL_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
SAFE_NAME_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}")

REQUEST_KEYS = {
    "audio_ms",
    "cache_keys",
    "deadline_provenance",
    "deadline_us",
    "event_id",
    "images",
    "input_tokens",
    "model_class",
    "observed_latency_us",
    "output_tokens",
    "priority_class",
    "priority_provenance",
    "provenance",
    "retrieved_chunks",
    "schema_version",
    "service",
    "session_id",
    "source",
    "source_fields",
    "t_us",
}
SOURCE_FIELD_KEYS = {
    "burstgpt_failed",
    "elapsed_time",
    "log_type",
    "model",
    "total_tokens",
}


class ReplayError(RuntimeError):
    pass


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def require_int(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    require(is_int(value), f"{field}: expected integer")
    if minimum is not None:
        require(value >= minimum, f"{field}: expected >= {minimum}")
    if maximum is not None:
        require(value <= maximum, f"{field}: expected <= {maximum}")
    return value


def require_string(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(value), f"{field}: expected string")
    return value


def require_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field}: expected object")
    actual = set(value)
    require(
        actual == expected,
        f"{field}: keys differ; missing={sorted(expected - actual)}, "
        f"unknown={sorted(actual - expected)}",
    )
    return value


def reject_constant(value: str) -> None:
    raise ReplayError(f"invalid JSON constant {value}")


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReplayError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json_bytes(raw: bytes, field: str) -> Any:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ReplayError(f"{field}: expected ASCII JSON") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ReplayError(f"{field}: invalid JSON: {error}") from error


def read_json(path: Path, field: str) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ReplayError(f"{field}: cannot read {path}: {error}") from error
    return parse_json_bytes(raw, field), raw


def canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ReplayError(f"cannot encode canonical JSON: {error}") from error
    return (text + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def checked_sha256(value: Any, field: str) -> str:
    digest = require_string(value, field)
    require(SHA256_RE.fullmatch(digest) is not None, f"{field}: invalid SHA-256")
    return digest


def verify_local_digest(path: Path, expected: str, field: str) -> None:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ReplayError(f"{field}: cannot read {path}: {error}") from error
    require(sha256_bytes(raw) == expected, f"{field}: local SHA-256 mismatch")


def safe_child(root: Path, name: Any, field: str) -> Path:
    name = require_string(name, field)
    require(SAFE_NAME_RE.fullmatch(name) is not None, f"{field}: unsafe name")
    require(name not in {".", ".."}, f"{field}: unsafe name")
    path = (root / name).resolve()
    require(path.parent == root.resolve(), f"{field}: escapes root")
    return path


def verify_output(
    bundle_dir: Path,
    record: Any,
    field: str,
) -> tuple[Path, bytes]:
    record = require_keys(record, {"bytes", "path", "records", "sha256"}, field)
    path = safe_child(bundle_dir, record["path"], f"{field}.path")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ReplayError(f"{field}: cannot read {path}: {error}") from error
    expected_bytes = require_int(record["bytes"], f"{field}.bytes", minimum=0)
    expected_records = require_int(record["records"], f"{field}.records", minimum=0)
    expected_sha256 = checked_sha256(record["sha256"], f"{field}.sha256")
    require(len(raw) == expected_bytes, f"{field}: byte count mismatch")
    require(sha256_bytes(raw) == expected_sha256, f"{field}: SHA-256 mismatch")
    if path.suffix == ".jsonl":
        require(raw.endswith(b"\n"), f"{field}: JSONL lacks final newline")
        require(raw.count(b"\n") == expected_records, f"{field}: record mismatch")
    else:
        require(expected_records == 1, f"{field}: JSON record count must be one")
    return path, raw


def validate_selector(selector: Any) -> dict[str, Any]:
    selector = require_keys(
        selector,
        {
            "active_bundle",
            "active_profile",
            "fallback_bundle",
            "fallback_profile",
            "schema_version",
        },
        "selector",
    )
    require_int(selector["schema_version"], "selector.schema_version")
    require(selector["schema_version"] == 1, "selector: unsupported schema_version")
    for field in (
        "active_bundle",
        "active_profile",
        "fallback_bundle",
        "fallback_profile",
    ):
        value = require_string(selector[field], f"selector.{field}")
        require(SAFE_NAME_RE.fullmatch(value) is not None, f"selector.{field}: unsafe")
        require(value not in {".", ".."}, f"selector.{field}: unsafe")
    require(
        selector["active_bundle"] != selector["fallback_bundle"],
        "selector: active and fallback bundles must differ",
    )
    return selector


def validate_manifest(manifest: Any, active_profile: str) -> dict[str, Any]:
    manifest = require_keys(
        manifest,
        {
            "builder_sha256",
            "builder_version",
            "limitations",
            "model_assignment_provenance",
            "outputs",
            "profile",
            "s8_config_sha256",
            "s8_normalizer_sha256",
            "schema_version",
            "selection",
            "shared_builder_sha256",
            "source",
            "statistics",
            "trace_provenance",
        },
        "manifest",
    )
    require_int(manifest["schema_version"], "manifest.schema_version")
    require(manifest["schema_version"] == 1, "manifest: unsupported schema_version")
    require(manifest["profile"] == active_profile, "manifest: active profile mismatch")
    require(manifest["trace_provenance"] == "real", "manifest: trace not real")
    require(
        manifest["model_assignment_provenance"] == "semi_synthetic",
        "manifest: assignment provenance mismatch",
    )
    for field in (
        "builder_sha256",
        "shared_builder_sha256",
        "s8_config_sha256",
        "s8_normalizer_sha256",
    ):
        checked_sha256(manifest[field], f"manifest.{field}")
    outputs = require_keys(
        manifest["outputs"],
        {"model_assignment", "requests"},
        "manifest.outputs",
    )
    selection = manifest["selection"]
    require(isinstance(selection, dict), "manifest.selection: expected object")
    for field in ("window_us", "source_row_first", "source_row_last"):
        require_int(selection.get(field), f"manifest.selection.{field}", minimum=0)
    constraints = require_keys(
        selection.get("constraints"),
        {
            "cold_trigger_gap_us",
            "hot_strictly_exceeds_cold",
            "maximum_offered_tokens",
            "maximum_successful_requests",
            "minimum_cold_request_tokens",
            "minimum_target_dwell_us",
        },
        "manifest.selection.constraints",
    )
    for field in (
        "cold_trigger_gap_us",
        "maximum_offered_tokens",
        "maximum_successful_requests",
        "minimum_target_dwell_us",
        "minimum_cold_request_tokens",
    ):
        require_int(
            constraints.get(field),
            f"manifest.selection.constraints.{field}",
            minimum=0,
        )
    require(
        constraints["hot_strictly_exceeds_cold"] is True,
        "manifest.selection.constraints.hot_strictly_exceeds_cold: expected true",
    )
    require(isinstance(manifest["statistics"], dict), "manifest.statistics: expected object")
    return manifest


def validate_assignment(
    assignment: Any,
    *,
    profile: str,
    request_sha256: str,
    window_us: int,
    shard_manifest: Any,
) -> tuple[dict[str, Any], str, str]:
    assignment = require_keys(
        assignment,
        {
            "execution_contract",
            "mapping_rule",
            "mappings",
            "profile",
            "provenance",
            "schema_version",
            "source_trace_sha256",
        },
        "assignment",
    )
    require_int(assignment["schema_version"], "assignment.schema_version")
    require(assignment["schema_version"] == 1, "assignment: unsupported schema_version")
    require(assignment["profile"] == profile, "assignment: profile mismatch")
    require(assignment["provenance"] == "semi_synthetic", "assignment: provenance")
    require(
        assignment["mapping_rule"] == "exact source_fields.model lookup",
        "assignment: unsupported mapping rule",
    )
    require(
        checked_sha256(assignment["source_trace_sha256"], "assignment.source_trace_sha256")
        == request_sha256,
        "assignment: request artifact is not bound",
    )
    contract = require_keys(
        assignment["execution_contract"],
        {"deadline", "duration_us", "failed_source_requests", "payload", "priority"},
        "assignment.execution_contract",
    )
    require_int(contract["duration_us"], "assignment.duration_us", minimum=1)
    require(contract["duration_us"] == window_us, "assignment: duration mismatch")
    require(contract["failed_source_requests"] == "skip", "assignment: failed policy")
    require(contract["deadline"] == "unset", "assignment: deadline must be unset")
    require(contract["priority"] == "unset", "assignment: priority must be unset")

    mappings = assignment["mappings"]
    require(isinstance(mappings, dict) and len(mappings) == 2, "assignment: need two models")
    shard_manifest = require_keys(
        shard_manifest,
        {
            "adb_server_port",
            "created_utc",
            "models",
            "readiness",
            "readiness_reason",
            "remote_root",
            "schema_version",
            "sharder",
        },
        "shard_manifest",
    )
    require_int(shard_manifest["schema_version"], "shard_manifest.schema_version")
    require(shard_manifest["schema_version"] == 1, "shard_manifest: schema_version")
    shard_models = shard_manifest["models"]
    require(isinstance(shard_models, dict), "shard_manifest.models: expected object")

    model_ids = set()
    role_to_id = {}
    for source_model, mapping in mappings.items():
        require_string(source_model, "assignment source model")
        mapping = require_keys(
            mapping,
            {
                "artifact",
                "model_id",
                "planned_initial_placement",
                "role",
            },
            f"assignment.mappings.{source_model}",
        )
        model_id = require_string(mapping["model_id"], f"{source_model}.model_id")
        require(MODEL_ID_RE.fullmatch(model_id) is not None, f"{source_model}: model_id")
        require(model_id not in model_ids, f"assignment: duplicate model_id {model_id}")
        model_ids.add(model_id)
        role = mapping["role"]
        placement = mapping["planned_initial_placement"]
        require(role in {"hot", "cold"}, f"{source_model}: invalid role")
        require(role not in role_to_id, f"assignment: duplicate role {role}")
        if role == "hot":
            require(placement == "server_cuda", f"{source_model}: hot placement")
        else:
            require(placement == "phone_resident", f"{source_model}: cold placement")
        role_to_id[role] = model_id

        artifact = require_keys(
            mapping["artifact"],
            {"bytes", "file_name", "sha256"},
            f"{source_model}.artifact",
        )
        artifact_record = {
            "bytes": require_int(artifact["bytes"], f"{source_model}.artifact.bytes", minimum=1),
            "file_name": require_string(
                artifact["file_name"], f"{source_model}.artifact.file_name"
            ),
            "sha256": checked_sha256(
                artifact["sha256"], f"{source_model}.artifact.sha256"
            ),
        }
        require(model_id in shard_models, f"{source_model}: model missing from shard manifest")
        shard_model = shard_models[model_id]
        require(isinstance(shard_model, dict), f"shard_manifest.models.{model_id}")
        source = shard_model.get("source")
        source = require_keys(
            source,
            {"architecture", "block_count", "bytes", "file_name", "sha256"},
            f"shard_manifest.models.{model_id}.source",
        )
        shard_bytes = require_int(
            source["bytes"],
            f"shard_manifest.models.{model_id}.source.bytes",
            minimum=1,
        )
        shard_file_name = require_string(
            source["file_name"],
            f"shard_manifest.models.{model_id}.source.file_name",
        )
        shard_sha256 = checked_sha256(
            source["sha256"],
            f"shard_manifest.models.{model_id}.source.sha256",
        )
        require_string(
            source["architecture"],
            f"shard_manifest.models.{model_id}.source.architecture",
        )
        require_int(
            source["block_count"],
            f"shard_manifest.models.{model_id}.source.block_count",
            minimum=1,
        )
        shard_artifact = {
            "bytes": shard_bytes,
            "file_name": shard_file_name,
            "sha256": shard_sha256,
        }
        require(
            canonical_bytes(artifact_record) == canonical_bytes(shard_artifact),
            f"{source_model}: artifact is not bound to shard manifest",
        )

    require(set(shard_models) == model_ids, "shard_manifest: unbound canonical model")
    require(set(role_to_id) == {"hot", "cold"}, "assignment: missing hot/cold role")
    return mappings, role_to_id["hot"], role_to_id["cold"]


def parse_requests(
    raw: bytes,
    *,
    mappings: dict[str, Any],
    duration_us: int,
) -> list[dict[str, Any]]:
    require(raw.endswith(b"\n"), "requests: missing final newline")
    records = []
    previous_key: tuple[int, int] | None = None
    event_ids = set()
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        require(line.endswith(b"\n"), f"requests:{line_number}: incomplete line")
        record = parse_json_bytes(line, f"requests:{line_number}")
        require(
            canonical_bytes(record) == line,
            f"requests:{line_number}: record is not canonical JSON",
        )
        record = require_keys(record, REQUEST_KEYS, f"requests:{line_number}")
        require_int(record["schema_version"], f"requests:{line_number}.schema_version")
        require(record["schema_version"] == 1, f"requests:{line_number}: schema_version")
        event_id = require_string(record["event_id"], f"requests:{line_number}.event_id")
        source = require_string(record["source"], f"requests:{line_number}.source")
        prefix = source + ":"
        require(event_id.startswith(prefix), f"requests:{line_number}: event/source mismatch")
        suffix = event_id[len(prefix) :]
        require(
            suffix.isascii() and suffix.isdigit() and len(suffix) <= 20,
            f"requests:{line_number}: event suffix",
        )
        source_row = int(suffix)
        t_us = require_int(
            record["t_us"],
            f"requests:{line_number}.t_us",
            minimum=0,
            maximum=duration_us - 1,
        )
        key = (t_us, source_row)
        require(
            previous_key is None or key > previous_key,
            f"requests:{line_number}: nonmonotonic order",
        )
        previous_key = key
        require(event_id not in event_ids, f"requests:{line_number}: duplicate event_id")
        event_ids.add(event_id)

        require(record["provenance"] == "real", f"requests:{line_number}: provenance")
        for field in (
            "input_tokens",
            "output_tokens",
            "images",
            "audio_ms",
            "retrieved_chunks",
        ):
            require_int(record[field], f"requests:{line_number}.{field}", minimum=0)
        require(record["observed_latency_us"] is None, f"requests:{line_number}: latency")
        require(record["priority_class"] is None, f"requests:{line_number}: priority")
        require(record["deadline_us"] is None, f"requests:{line_number}: deadline")
        require(
            record["priority_provenance"] == "none",
            f"requests:{line_number}: priority provenance",
        )
        require(
            record["deadline_provenance"] == "none",
            f"requests:{line_number}: deadline provenance",
        )
        require(
            record["session_id"] is None or isinstance(record["session_id"], str),
            f"requests:{line_number}: session_id",
        )
        require(
            isinstance(record["cache_keys"], list)
            and all(isinstance(value, str) for value in record["cache_keys"]),
            f"requests:{line_number}: cache_keys",
        )
        require_string(record["service"], f"requests:{line_number}.service")
        require_string(record["model_class"], f"requests:{line_number}.model_class")

        source_fields = require_keys(
            record["source_fields"],
            SOURCE_FIELD_KEYS,
            f"requests:{line_number}.source_fields",
        )
        source_model = require_string(
            source_fields["model"], f"requests:{line_number}.source_fields.model"
        )
        require(source_model in mappings, f"requests:{line_number}: unknown model")
        require(
            isinstance(source_fields["burstgpt_failed"], bool),
            f"requests:{line_number}: burstgpt_failed",
        )
        require_int(
            source_fields["elapsed_time"],
            f"requests:{line_number}.source_fields.elapsed_time",
            minimum=0,
        )
        require_string(
            source_fields["log_type"],
            f"requests:{line_number}.source_fields.log_type",
        )
        total_tokens = require_int(
            source_fields["total_tokens"],
            f"requests:{line_number}.source_fields.total_tokens",
            minimum=0,
        )
        require(
            total_tokens == record["input_tokens"] + record["output_tokens"],
            f"requests:{line_number}: token total mismatch",
        )
        require(
            source_fields["burstgpt_failed"] == (record["output_tokens"] == 0),
            f"requests:{line_number}: failure marker mismatch",
        )
        records.append(record)
    require(records, "requests: empty artifact")
    return records


def derive_cycles(
    successful: list[dict[str, Any]],
    *,
    cold_source_model: str,
    duration_us: int,
    trigger_gap_us: int,
    minimum_dwell_us: int,
    minimum_request_tokens: int,
) -> list[dict[str, Any]]:
    cycles = []
    index = 0
    while index < len(successful):
        source_model = successful[index]["source_fields"]["model"]
        end = index + 1
        while (
            end < len(successful)
            and successful[end]["source_fields"]["model"] == source_model
        ):
            end += 1
        run = successful[index:end]
        if source_model == cold_source_model:
            trigger_index = next(
                (
                    offset
                    for offset in range(1, len(run))
                    if run[offset]["t_us"] - run[offset - 1]["t_us"]
                    <= trigger_gap_us
                ),
                None,
            )
            if trigger_index is not None:
                return_record = successful[end] if end < len(successful) else None
                trigger_record = run[trigger_index]
                return_us = return_record["t_us"] if return_record else duration_us
                maximum_request_tokens = max(
                    row["input_tokens"] + row["output_tokens"] for row in run
                )
                dwell_us = return_us - trigger_record["t_us"]
                cycles.append(
                    {
                        "start_us": run[0]["t_us"],
                        "trigger_us": trigger_record["t_us"],
                        "last_cold_arrival_us": run[-1]["t_us"],
                        "return_us": return_us,
                        "has_return_request": return_record is not None,
                        "target_dwell_us": dwell_us,
                        "request_count": len(run),
                        "input_tokens": sum(row["input_tokens"] for row in run),
                        "output_tokens": sum(row["output_tokens"] for row in run),
                        "maximum_request_tokens": maximum_request_tokens,
                        "viable_30s": dwell_us >= minimum_dwell_us,
                        "substantial_request": maximum_request_tokens
                        >= minimum_request_tokens,
                        "useful_handoff_cycle": dwell_us >= minimum_dwell_us
                        and maximum_request_tokens >= minimum_request_tokens,
                        "_trigger_event_id": trigger_record["event_id"],
                        "_return_event_id": (
                            return_record["event_id"] if return_record else None
                        ),
                        "_source_event_ids": [row["event_id"] for row in run],
                    }
                )
        index = end
    return cycles


def public_cycle(cycle: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cycle.items() if not key.startswith("_")}


def validate_statistics(
    manifest: dict[str, Any],
    requests: list[dict[str, Any]],
    mappings: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    statistics = manifest["statistics"]
    selection = manifest["selection"]
    constraints = selection["constraints"]
    successful = [row for row in requests if row["output_tokens"] > 0]
    failed = len(requests) - len(successful)
    source_counts = {
        source_model: sum(
            row["source_fields"]["model"] == source_model for row in successful
        )
        for source_model in sorted(mappings)
    }
    cold_sources = [
        source_model
        for source_model, mapping in mappings.items()
        if mapping["role"] == "cold"
    ]
    require(len(cold_sources) == 1, "assignment: expected one cold source model")
    cold_source = cold_sources[0]
    cycles = derive_cycles(
        successful,
        cold_source_model=cold_source,
        duration_us=selection["window_us"],
        trigger_gap_us=constraints["cold_trigger_gap_us"],
        minimum_dwell_us=constraints["minimum_target_dwell_us"],
        minimum_request_tokens=constraints["minimum_cold_request_tokens"],
    )
    transitions = sum(
        left["source_fields"]["model"] != right["source_fields"]["model"]
        for left, right in zip(successful, successful[1:])
    )
    expected = {
        "source_rows": len(requests),
        "successful_requests": len(successful),
        "failed_requests": failed,
        "successful_model_counts": source_counts,
        "successful_input_tokens": sum(row["input_tokens"] for row in successful),
        "successful_output_tokens": sum(row["output_tokens"] for row in successful),
        "successful_offered_tokens": sum(
            row["input_tokens"] + row["output_tokens"] for row in successful
        ),
        "successful_model_transitions": transitions,
        "qualifying_switch_cycles": len(cycles),
        "viable_30s_cycles": sum(cycle["viable_30s"] for cycle in cycles),
        "useful_handoff_cycles": sum(
            cycle["useful_handoff_cycle"] for cycle in cycles
        ),
        "target_switches_during_arrivals": len(cycles)
        + sum(cycle["has_return_request"] for cycle in cycles),
        "switch_cycles": [public_cycle(cycle) for cycle in cycles],
        "first_arrival_us": requests[0]["t_us"],
        "last_arrival_us": requests[-1]["t_us"],
    }
    require(
        canonical_bytes(statistics) == canonical_bytes(expected),
        "manifest: statistics do not match request trace",
    )
    first_row = int(requests[0]["event_id"].rsplit(":", 1)[1])
    last_row = int(requests[-1]["event_id"].rsplit(":", 1)[1])
    require(selection["source_row_first"] == first_row, "manifest: first row mismatch")
    require(selection["source_row_last"] == last_row, "manifest: last row mismatch")
    return cycles, cold_source


def make_intents(
    cycles: list[dict[str, Any]],
    *,
    hot_model_id: str,
    cold_model_id: str,
) -> list[dict[str, Any]]:
    intents = []
    for cycle_index, cycle in enumerate(cycles):
        window_seed = {
            "cold_model_id": cold_model_id,
            "return_us": cycle["return_us"],
            "source_event_ids": cycle["_source_event_ids"],
            "trigger_event_id": cycle["_trigger_event_id"],
            "trigger_us": cycle["trigger_us"],
        }
        window_id = "s39-window-v1:" + sha256_bytes(canonical_bytes(window_seed))
        candidates = [
            {
                "cycle_index": cycle_index,
                "from_model_id": hot_model_id,
                "kind": "PROMOTE",
                "source_event_id": cycle["_trigger_event_id"],
                "t_us": cycle["trigger_us"],
                "to_model_id": cold_model_id,
                "window_id": window_id,
            }
        ]
        if cycle["has_return_request"]:
            candidates.append(
                {
                    "cycle_index": cycle_index,
                    "from_model_id": cold_model_id,
                    "kind": "DEMOTE",
                    "source_event_id": cycle["_return_event_id"],
                    "t_us": cycle["return_us"],
                    "to_model_id": hot_model_id,
                    "window_id": window_id,
                }
            )
        for candidate in candidates:
            sequence = len(intents)
            record = {
                "schema_version": 1,
                "sequence": sequence,
                **candidate,
            }
            record["intent_id"] = (
                f"s39-intent-v1:{sequence:04d}:"
                + sha256_bytes(canonical_bytes(record))
            )
            intents.append(record)
    require(
        all(left["t_us"] < right["t_us"] for left, right in zip(intents, intents[1:])),
        "derived intents are not strictly ordered",
    )
    expected_from = hot_model_id
    for intent in intents:
        require(intent["from_model_id"] == expected_from, "intent target discontinuity")
        expected_from = intent["to_model_id"]
    return intents


def atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise ReplayError(f"cannot write {path}: {error}") from error


def build(
    *,
    root: Path,
    selector_path: Path,
    shard_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    root = root.resolve()
    selector_path = selector_path.resolve()
    shard_manifest_path = shard_manifest_path.resolve()
    selector, selector_raw = read_json(selector_path, "selector")
    selector = validate_selector(selector)
    bundle_dir = safe_child(root, selector["active_bundle"], "selector.active_bundle")
    require(bundle_dir.is_dir(), "selector: active bundle is not a directory")

    manifest_path = bundle_dir / "manifest.json"
    manifest, manifest_raw = read_json(manifest_path, "manifest")
    manifest = validate_manifest(manifest, selector["active_profile"])
    require(
        manifest["builder_version"] == "s39-phone-model-switch-frequent-trace-v1",
        "manifest: unsupported builder_version",
    )
    verify_local_digest(
        root / "build_frequent_trace.py",
        manifest["builder_sha256"],
        "manifest.builder_sha256",
    )
    verify_local_digest(
        root / "build_trace.py",
        manifest["shared_builder_sha256"],
        "manifest.shared_builder_sha256",
    )
    verify_local_digest(
        root.parent / "s8_operator_island_affinity" / "normalize_trace.py",
        manifest["s8_normalizer_sha256"],
        "manifest.s8_normalizer_sha256",
    )
    verify_local_digest(
        root.parent
        / "s8_operator_island_affinity"
        / "configs"
        / "burstgpt.config.json",
        manifest["s8_config_sha256"],
        "manifest.s8_config_sha256",
    )
    _, requests_raw = verify_output(
        bundle_dir,
        manifest["outputs"]["requests"],
        "manifest.outputs.requests",
    )
    _, assignment_raw = verify_output(
        bundle_dir,
        manifest["outputs"]["model_assignment"],
        "manifest.outputs.model_assignment",
    )
    assignment = parse_json_bytes(assignment_raw, "assignment")
    require(
        canonical_bytes(assignment) == assignment_raw,
        "assignment: not canonical JSON",
    )
    shard_manifest, shard_manifest_raw = read_json(
        shard_manifest_path,
        "shard_manifest",
    )
    mappings, hot_model_id, cold_model_id = validate_assignment(
        assignment,
        profile=selector["active_profile"],
        request_sha256=sha256_bytes(requests_raw),
        window_us=manifest["selection"]["window_us"],
        shard_manifest=shard_manifest,
    )
    requests = parse_requests(
        requests_raw,
        mappings=mappings,
        duration_us=manifest["selection"]["window_us"],
    )
    require(
        len(requests) == manifest["outputs"]["requests"]["records"],
        "requests: manifest record count mismatch",
    )
    require(
        set(row["source_fields"]["model"] for row in requests) == set(mappings),
        "assignment: mapping without trace records",
    )
    cycles, _ = validate_statistics(manifest, requests, mappings)
    intents = make_intents(
        cycles,
        hot_model_id=hot_model_id,
        cold_model_id=cold_model_id,
    )
    require(
        len(intents) == manifest["statistics"]["target_switches_during_arrivals"],
        "intent count does not match declared target changes",
    )
    intents_raw = b"".join(canonical_bytes(record) for record in intents)
    intents_path = output_dir / "replay_intents.jsonl"
    replay_manifest = {
        "schema_version": 1,
        "builder_version": VERSION,
        "active_bundle": selector["active_bundle"],
        "active_profile": selector["active_profile"],
        "canonical_models": {
            "cold": cold_model_id,
            "hot": hot_model_id,
        },
        "inputs": {
            "active_trace_sha256": sha256_bytes(selector_raw),
            "bundle_manifest_sha256": sha256_bytes(manifest_raw),
            "model_assignment_sha256": sha256_bytes(assignment_raw),
            "replay_builder_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "requests_sha256": sha256_bytes(requests_raw),
            "shard_manifest_sha256": sha256_bytes(shard_manifest_raw),
        },
        "policy": {
            "cold_trigger_gap_us": manifest["selection"]["constraints"][
                "cold_trigger_gap_us"
            ],
            "minimum_request_tokens": manifest["selection"]["constraints"][
                "minimum_cold_request_tokens"
            ],
            "minimum_target_dwell_us": manifest["selection"]["constraints"][
                "minimum_target_dwell_us"
            ],
            "source": "bundle manifest plus ordered successful requests",
        },
        "derived": {
            "first_intent_us": intents[0]["t_us"] if intents else None,
            "last_intent_us": intents[-1]["t_us"] if intents else None,
            "promotion_windows": len(cycles),
            "target_changes": len(intents),
            "trace_horizon_us": manifest["selection"]["window_us"],
        },
        "output": {
            "bytes": len(intents_raw),
            "path": intents_path.name,
            "records": len(intents),
            "sha256": sha256_bytes(intents_raw),
        },
        "semantics": "policy_intents_only_no_readiness_or_execution_claim",
    }
    atomic_write(intents_path, intents_raw)
    atomic_write(output_dir / "replay_manifest.json", canonical_bytes(replay_manifest))
    return replay_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fail-closed S39 promotion and demotion intents"
    )
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--selector", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument(
        "--shard-manifest",
        type=Path,
        default=DEFAULT_SHARD_MANIFEST,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        result = build(
            root=arguments.root,
            selector_path=arguments.selector,
            shard_manifest_path=arguments.shard_manifest,
            output_dir=arguments.output,
        )
    except (ReplayError, OSError) as error:
        raise SystemExit(f"S39_REPLAY_ERROR: {error}") from None
    print(json.dumps(result["derived"], sort_keys=True))
