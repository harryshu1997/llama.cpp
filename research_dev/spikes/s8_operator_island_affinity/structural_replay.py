#!/usr/bin/python3
"""Fail-closed structural replay for normalized S8 Gate-A traces."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError as exc:
    raise SystemExit(f"E_DEPENDENCY: jsonschema 4.10.3 is required: {exc}")


HERE = Path(__file__).resolve().parent
SAFE_MAX = 2**53 - 1
MAX_RECORDS = 20_000
MAX_TRACE_BYTES = 512 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_ARTIFACT_OUTPUTS = 16
MAX_DAG_FILES = 32
MAX_DAG_NODES = 64
MAX_DAG_DEPENDENCIES = 256
SCENARIOS = ("low", "median", "high", "burst")
DEMAND_FIELDS = (
    "input_tokens",
    "output_tokens",
    "images",
    "audio_ms",
    "retrieved_chunks",
)
TRACE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.jsonl$")
CANONICAL_UINT_RE = re.compile(r"^(0|[1-9][0-9]*)$", re.ASCII)


def _is_direct_source_execution() -> bool:
    try:
        module_path = Path(__file__).resolve()
        argv_path = Path(sys.argv[0]).resolve()
    except OSError:
        return False
    return (
        __name__ == "__main__"
        and __spec__ is None
        and globals().get("__cached__") is None
        and module_path.suffix == ".py"
        and argv_path == module_path
    )


DIRECT_SOURCE_EXECUTION = _is_direct_source_execution()


class ReplayError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def fail(code: str, message: str) -> None:
    raise ReplayError(code, message)


def _reject_constant(value: str) -> None:
    fail("E_JSON", f"non-finite JSON number is forbidden: {value}")


def _reject_float(value: str) -> None:
    fail("E_JSON", f"JSON float is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("E_JSON_DUPLICATE_KEY", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data: bytes, label: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail("E_UTF8", f"{label}: invalid UTF-8 at byte {exc.start}")
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except ReplayError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        fail("E_JSON", f"{label}: invalid JSON: {exc}")


def _check_json_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)) or type(value) is int:
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                fail("E_CANONICAL", f"{path}: object key is not a string")
            _check_json_value(item, f"{path}.{key}")
        return
    fail("E_CANONICAL", f"{path}: unsupported value type {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    _check_json_value(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        fail("E_CANONICAL", f"cannot serialize JSON: {exc}")


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def checked_add(name: str, left: int, right: int) -> int:
    value = left + right
    if value > SAFE_MAX:
        fail("E_DEMAND_OVERFLOW", f"{name} exceeds the safe integer bound")
    return value


def normalizer_bundle_hash() -> str:
    path = HERE / "normalize_trace.py"
    digest = hashlib.sha256(read_once(path, MAX_METADATA_BYTES)).hexdigest()
    return sha256(f"normalize_trace.py  {digest}".encode("ascii"))


def read_once(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        fail("E_IO", f"cannot open {path}: {exc}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail("E_IO", f"not a regular file: {path}")
        if before.st_size > limit:
            fail("E_SIZE_LIMIT", f"{path}: {before.st_size} bytes exceeds {limit}")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                fail("E_IO", f"short read from {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            fail("E_FILE_CHANGED", f"{path}: file grew while reading")
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            fail("E_FILE_CHANGED", f"{path}: file changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


STARTUP_REPLAY_CODE_SHA256 = sha256(
    read_once(Path(__file__).resolve(), MAX_METADATA_BYTES)
)


def require_jsonschema() -> None:
    try:
        version = importlib.metadata.version("jsonschema")
    except importlib.metadata.PackageNotFoundError:
        fail("E_DEPENDENCY", "jsonschema package is absent")
    if version != "4.10.3":
        fail("E_DEPENDENCY", f"jsonschema {version} != pinned 4.10.3")


def load_validator(name: str) -> jsonschema.Draft202012Validator:
    schema_path = HERE / "schemas" / name
    schema = parse_json(read_once(schema_path, MAX_METADATA_BYTES), str(schema_path))
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        return jsonschema.Draft202012Validator(schema)
    except jsonschema.SchemaError as exc:
        fail("E_SCHEMA", f"invalid schema {name}: {exc.message}")


def validate_schema(
    validator: jsonschema.Draft202012Validator,
    value: Any,
    label: str,
) -> None:
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        error = errors[0]
        path = "$" + "".join(f"[{item!r}]" for item in error.absolute_path)
        fail("E_SCHEMA", f"{label}{path}: {error.message}")


def load_dags(
    dag_dir: Path,
    validator: jsonschema.Draft202012Validator,
) -> dict[str, tuple[dict[str, Any], str]]:
    if not dag_dir.is_dir():
        fail("E_DAG", f"DAG directory does not exist: {dag_dir}")
    paths = sorted(dag_dir.glob("*.json"))
    if not paths:
        fail("E_DAG", f"DAG directory is empty: {dag_dir}")
    if len(paths) > MAX_DAG_FILES:
        fail("E_DAG_LIMIT", f"DAG catalog exceeds {MAX_DAG_FILES} files")
    result: dict[str, tuple[dict[str, Any], str]] = {}
    for path in paths:
        data = read_once(path, MAX_METADATA_BYTES)
        dag = parse_json(data, str(path))
        validate_schema(validator, dag, str(path))
        validate_dag_semantics(dag, str(path))
        service = dag["service"]
        if service in result:
            fail("E_DAG", f"duplicate DAG service: {service}")
        result[service] = (dag, sha256(canonical_json(dag)))
    return result


def validate_dag_semantics(dag: dict[str, Any], label: str) -> None:
    if len(dag["nodes"]) > MAX_DAG_NODES:
        fail("E_DAG_LIMIT", f"{label}: exceeds {MAX_DAG_NODES} nodes")
    node_ids = [node["node_id"] for node in dag["nodes"]]
    if len(set(node_ids)) != len(node_ids):
        fail("E_DAG", f"{label}: duplicate node_id")
    known = set(node_ids)
    dependency_count = 0
    for node in dag["nodes"]:
        deps = node["depends_on"]
        dependency_count += len(deps)
        if len(set(deps)) != len(deps):
            fail("E_DAG", f"{label}: duplicate dependency for {node['node_id']}")
        missing = sorted(set(deps) - known)
        if missing:
            fail("E_DAG", f"{label}: missing dependencies {missing}")
    if dependency_count > MAX_DAG_DEPENDENCIES:
        fail(
            "E_DAG_LIMIT",
            f"{label}: exceeds {MAX_DAG_DEPENDENCIES} dependency edges",
        )

    state: dict[str, int] = {node_id: 0 for node_id in node_ids}
    dependencies = {node["node_id"]: node["depends_on"] for node in dag["nodes"]}

    def visit(node_id: str) -> None:
        if state[node_id] == 1:
            fail("E_DAG", f"{label}: dependency cycle at {node_id}")
        if state[node_id] == 2:
            return
        state[node_id] = 1
        for dependency in dependencies[node_id]:
            visit(dependency)
        state[node_id] = 2

    for node_id in node_ids:
        visit(node_id)


def parse_trace(
    trace_bytes: bytes,
    validator: jsonschema.Draft202012Validator,
    config: dict[str, Any],
    sidecar: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], tuple[int, int, int], tuple[int, int, int]]:
    if not trace_bytes:
        fail("E_EMPTY_TRACE", "normalized trace is empty")
    if b"\r" in trace_bytes or not trace_bytes.endswith(b"\n"):
        fail("E_TRACE_FORMAT", "trace must use LF and end with one LF")
    if trace_bytes.endswith(b"\n\n"):
        fail("E_TRACE_FORMAT", "trace has a blank terminal line")
    if trace_bytes.count(b"\n") > MAX_RECORDS:
        fail("E_RECORD_LIMIT", f"trace exceeds {MAX_RECORDS} records")

    raw_lines = trace_bytes[:-1].split(b"\n")
    if not raw_lines or any(line == b"" for line in raw_lines):
        fail("E_EMPTY_TRACE", "trace contains no records or a blank line")
    records = []
    seen_ids: set[str] = set()
    previous_key: tuple[int, int, int] | None = None
    first_key: tuple[int, int, int] | None = None
    service_totals: dict[str, dict[str, int]] = {}
    for index, line in enumerate(raw_lines):
        if len(line) > MAX_LINE_BYTES:
            fail("E_SIZE_LIMIT", f"trace line {index + 1} exceeds {MAX_LINE_BYTES} bytes")
        record = parse_json(line, f"trace line {index + 1}")
        if canonical_json(record) != line:
            fail("E_TRACE_FORMAT", f"trace line {index + 1} is not canonical JSON")
        validate_schema(validator, record, f"trace line {index + 1}")
        event_id = record["event_id"]
        if event_id in seen_ids:
            fail("E_DUPLICATE_EVENT_ID", f"duplicate event_id: {event_id}")
        seen_ids.add(event_id)
        if record["source"] != config["source"]:
            fail("E_SOURCE_PIN", f"{event_id}: source does not match source config")
        if (
            record["provenance"] != sidecar["provenance"]
            or record["provenance"] != config["provenance"]
        ):
            fail("E_SOURCE_PIN", f"{event_id}: provenance does not match sidecar")
        for key, expected in config["gate_a"].items():
            if record[key] != expected:
                fail("E_GATE_A_FIELDS", f"{event_id}: {key} differs from source config")
        if (
            record["deadline_us"] is not None
            or record["priority_class"] is not None
            or record["deadline_provenance"] != "none"
            or record["priority_provenance"] != "none"
            or record["observed_latency_us"] is not None
        ):
            fail("E_GATE_A_FIELDS", f"{event_id}: non-structural Gate-A fields present")
        window_span = (
            sidecar["window"]["t_end_us"] - sidecar["window"]["t_start_us"]
        )
        if record["t_us"] >= window_span:
            fail("E_ORDER", f"{event_id}: timestamp is outside the selected window")
        key = order_key(record)
        if previous_key is not None and key <= previous_key:
            fail("E_ORDER", f"{event_id}: total-order key {key} <= {previous_key}")
        if first_key is None:
            first_key = key
        previous_key = key
        totals = service_totals.setdefault(
            record["service"],
            {"request_count": 0, **{field: 0 for field in DEMAND_FIELDS}},
        )
        totals["request_count"] += 1
        for field in DEMAND_FIELDS:
            totals[field] = checked_add(
                f"{record['service']}.{field}",
                totals[field],
                record[field],
            )
        records.append(record)
    assert first_key is not None and previous_key is not None
    return records, service_totals, first_key, previous_key


def validate_records_against_window(
    records: list[dict[str, Any]],
    sidecar: dict[str, Any],
) -> None:
    window = sidecar["window"]
    source_rows = [order_key(record)[2] for record in records]
    if (
        min(source_rows) != window["source_row_first"]
        or max(source_rows) != window["source_row_last"]
    ):
        fail("E_ROW_RANGE", "trace source-row range does not match sidecar")
    offered_tokens = checked_add(
        "window.metric_value",
        sum(record["input_tokens"] for record in records),
        sum(record["output_tokens"] for record in records),
    )
    if offered_tokens != window["metric_value"]:
        fail("E_MANIFEST_SEMANTICS", "window metric_value does not match offered tokens")


def order_key(record: dict[str, Any]) -> tuple[int, int, int]:
    event_id = record["event_id"]
    if record["provenance"] == "semi_synthetic":
        parts = event_id.split(":", 3)
        if (
            len(parts) != 4
            or parts[0] != "mix"
            or not CANONICAL_UINT_RE.fullmatch(parts[1])
            or not parts[2]
        ):
            fail("E_EVENT_ID", f"invalid mixed event_id: {event_id}")
        rank = int(parts[1])
        row_text = parts[3]
    else:
        prefix = record["source"] + ":"
        if not event_id.startswith(prefix):
            fail("E_EVENT_ID", f"event_id does not bind source: {event_id}")
        rank = 0
        row_text = event_id[len(prefix):]
    if not CANONICAL_UINT_RE.fullmatch(row_text):
        fail("E_EVENT_ID", f"event_id has non-integer source row: {event_id}")
    row_id = int(row_text)
    if row_id > SAFE_MAX or rank > SAFE_MAX:
        fail("E_EVENT_ID", f"event_id integer exceeds safe bound: {event_id}")
    return record["t_us"], rank, row_id


def validate_sidecar_semantics(
    sidecar: dict[str, Any],
    config: dict[str, Any],
    trace_name: str,
) -> None:
    if sidecar["provenance"] == "semi_synthetic":
        fail("E_MIX_UNSUPPORTED", "mixed composition is not implemented in Gate-A P1")
    if sidecar["provenance"] != config["provenance"]:
        fail("E_SOURCE_PIN", "sidecar provenance does not match source config")
    if sidecar["time_scale_num"] != 1 or sidecar["time_scale_den"] != 1:
        fail("E_MANIFEST_SEMANTICS", "real component time scale must equal 1/1")
    if sidecar.get("filters") != [] or sidecar.get("exclusion_counts") != {}:
        fail("E_MANIFEST_SEMANTICS", "P1 real components require empty filters")
    window = sidecar["window"]
    if window["rule"] != "aligned_15m_offered_tokens_nearest_rank_v1":
        fail("E_MANIFEST_SEMANTICS", "window rule is not the frozen Gate-A rule")
    bin_index = window["bin_index"]
    window_us = 900_000_000
    if window["t_start_us"] != bin_index * window_us:
        fail("E_MANIFEST_SEMANTICS", "window.t_start_us does not match bin_index")
    if window["t_end_us"] != (bin_index + 1) * window_us:
        fail("E_MANIFEST_SEMANTICS", "window.t_end_us does not match bin_index")
    if window["source_row_first"] > window["source_row_last"]:
        fail("E_MANIFEST_SEMANTICS", "window source row range is reversed")
    rank = window.get("quantile_rank")
    n_bins = window.get("n_nonempty_bins")
    if n_bins is None:
        fail("E_MANIFEST_SEMANTICS", "window n_nonempty_bins is required")
    if (
        "timestamp_policy" not in window
        or "input_nonmonotonic_pairs" not in window
        or window["timestamp_policy"] != config["timestamp"]["policy"]
    ):
        fail("E_MANIFEST_SEMANTICS", "window timestamp policy does not match config")
    if window["scenario"] == "burst" and rank is not None:
        fail("E_MANIFEST_SEMANTICS", "burst window must not carry quantile_rank")
    if window["scenario"] != "burst" and rank is None:
        fail("E_MANIFEST_SEMANTICS", "quantile window requires quantile_rank")
    quantiles = {"low": (1, 10), "median": (1, 2), "high": (9, 10)}
    if rank is not None:
        numerator, denominator = quantiles[window["scenario"]]
        expected_rank = (numerator * n_bins + denominator - 1) // denominator
        if rank != expected_rank:
            fail("E_MANIFEST_SEMANTICS", "window quantile rank is not the frozen rank")
    if (
        window.get("input_nonmonotonic_pairs", 0) > 0
        and window.get("timestamp_policy") != "sort_stable"
    ):
        fail("E_MANIFEST_SEMANTICS", "nonmonotonic input requires sort_stable")
    expected_name = f"{config['source']}.{window['scenario']}.jsonl"
    if trace_name != expected_name:
        fail("E_MANIFEST_SEMANTICS", "trace filename does not match source/scenario")
    record_count = config["origin"]["record_count"]
    if window["source_row_last"] >= record_count:
        fail("E_MANIFEST_SEMANTICS", "window source row exceeds source record count")


def validate_source_binding(
    source_bytes: bytes,
    config: dict[str, Any],
    sidecar: dict[str, Any],
) -> str:
    source_hash = sha256(source_bytes)
    origin = config["origin"]
    if len(source_bytes) != origin["bytes"] or source_hash != origin["sha256"]:
        fail("E_SOURCE_PIN", "raw source bytes do not match the pinned source config")
    for key, sidecar_key in (
        ("url", "source_url"),
        ("revision", "source_revision"),
        ("filename", "source_file"),
        ("bytes", "source_bytes"),
        ("sha256", "source_sha256"),
        ("license", "license"),
    ):
        if sidecar[sidecar_key] != origin[key]:
            fail("E_SOURCE_PIN", f"sidecar {sidecar_key} does not match source config")
    return source_hash


def validate_artifact_members(
    run_dir: Path,
    artifact: dict[str, Any],
    config: dict[str, Any],
    source_bytes: bytes,
    validators: dict[str, jsonschema.Draft202012Validator],
    dags: dict[str, tuple[dict[str, Any], str]],
) -> dict[str, tuple[bytes, dict[str, Any], bytes]]:
    outputs = artifact["outputs"]
    if not outputs or len(outputs) > MAX_ARTIFACT_OUTPUTS:
        fail("E_ARTIFACT_BINDING", "artifact output count is outside the fixed bound")
    paths = [output["path"] for output in outputs]
    if len(paths) != len(set(paths)) or paths != sorted(paths):
        fail("E_ARTIFACT_BINDING", "artifact output paths must be unique and sorted")

    members: dict[str, tuple[bytes, dict[str, Any], bytes]] = {}
    for index, output in enumerate(outputs):
        name = output["path"]
        if not TRACE_NAME_RE.fullmatch(name) or Path(name).name != name:
            fail("E_ARTIFACT_BINDING", f"unsafe artifact output path: {name}")
        if output.get("sidecar_manifest_sha256") is None:
            fail("E_ARTIFACT_BINDING", f"output {index} has no sidecar hash")
        trace_bytes = read_once(run_dir / name, MAX_TRACE_BYTES)
        if not trace_bytes:
            fail("E_EMPTY_TRACE", f"{name}: normalized trace is empty")
        sidecar_name = name.replace(".jsonl", ".manifest.json")
        sidecar_bytes = read_once(run_dir / sidecar_name, MAX_METADATA_BYTES)
        sidecar = parse_json(sidecar_bytes, sidecar_name)
        if sidecar_bytes != canonical_json(sidecar) + b"\n":
            fail("E_CANONICAL", f"{sidecar_name}: sidecar file is not canonical JSON")
        validate_schema(validators["sidecar"], sidecar, sidecar_name)
        validate_sidecar_semantics(sidecar, config, name)
        validate_source_binding(source_bytes, config, sidecar)
        if output["sha256"] != sha256(trace_bytes):
            fail("E_TRACE_HASH", f"{name}: artifact output hash mismatch")
        if sidecar["output_sha256"] != output["sha256"]:
            fail("E_TRACE_HASH", f"{name}: sidecar output hash mismatch")
        if output["sidecar_manifest_sha256"] != sha256(canonical_json(sidecar)):
            fail("E_SIDECAR_HASH", f"{name}: artifact sidecar hash mismatch")
        if sidecar["output_row_count"] != trace_bytes.count(b"\n"):
            fail("E_ROW_COUNT", f"{name}: sidecar row count mismatch")
        if sidecar["normalizer_version"] != artifact["code_version"]:
            fail("E_ARTIFACT_BINDING", f"{name}: normalizer version mismatch")
        if sidecar["normalization_config_hash"] != artifact["config_hash"]:
            fail("E_CONFIG_BINDING", f"{name}: normalization config mismatch")
        records, _, _, _ = parse_trace(
            trace_bytes,
            validators["request"],
            config,
            sidecar,
        )
        validate_records_against_window(records, sidecar)
        for record in records:
            service = record["service"]
            if service not in dags:
                fail("E_SERVICE_UNKNOWN", f"no static DAG for service: {service}")
            if dags[service][0]["provenance"] != sidecar["provenance"]:
                fail("E_DAG_MISMATCH", f"{service}: DAG provenance does not match trace")
        members[name] = (trace_bytes, sidecar, sidecar_bytes)

    replay_preimage = "\n".join(
        output["sha256"] for output in sorted(outputs, key=lambda item: item["path"])
    ).encode("ascii")
    if artifact["deterministic_replay_sha256"] != sha256(replay_preimage):
        fail("E_ARTIFACT_BINDING", "artifact deterministic replay hash mismatch")
    if artifact.get("seed") is not None:
        fail("E_ARTIFACT_BINDING", "normalization artifact seed must be null")
    scenario_order = ("low", "median", "high", "burst")
    present = {
        members[output["path"]][1]["window"]["scenario"] for output in outputs
    }
    scenarios = [scenario for scenario in scenario_order if scenario in present]
    if len(scenarios) != len(outputs):
        fail("E_ARTIFACT_BINDING", "artifact scenarios must be unique")
    run_preimage = canonical_json(
        {
            "source_sha256": config["origin"]["sha256"],
            "config_hash": artifact["config_hash"],
            "code_version": artifact["code_version"],
            "scenarios": scenarios,
        }
    )
    expected_run_id = "normalize-" + hashlib.sha256(run_preimage).hexdigest()[:24]
    if artifact["run_id"] != expected_run_id:
        fail("E_ARTIFACT_BINDING", "artifact run_id mismatch")
    return members


def bind_inputs(
    trace_name: str,
    trace_bytes: bytes,
    sidecar: dict[str, Any],
    artifact: dict[str, Any],
    config: dict[str, Any],
    source_bytes: bytes,
    config_path: Path,
) -> dict[str, str]:
    trace_hash = sha256(trace_bytes)
    sidecar_hash = sha256(canonical_json(sidecar))
    config_hash = sha256(canonical_json(config))
    source_hash = validate_source_binding(source_bytes, config, sidecar)
    if sidecar["normalization_config_hash"] != config_hash:
        fail("E_CONFIG_BINDING", "sidecar normalization config hash mismatch")
    if sidecar["output_sha256"] != trace_hash:
        fail("E_TRACE_HASH", "trace bytes do not match sidecar output hash")

    outputs = [output for output in artifact["outputs"] if output["path"] == trace_name]
    if len(outputs) != 1:
        fail("E_ARTIFACT_BINDING", f"artifact must bind trace exactly once: {trace_name}")
    output = outputs[0]
    if output["sha256"] != trace_hash:
        fail("E_TRACE_HASH", "trace bytes do not match artifact output hash")
    if output["sidecar_manifest_sha256"] != sidecar_hash:
        fail("E_SIDECAR_HASH", "sidecar does not match artifact sidecar hash")
    if artifact["code_version"] != sidecar["normalizer_version"]:
        fail("E_ARTIFACT_BINDING", "artifact and sidecar code versions differ")
    if artifact["config_hash"] != config_hash:
        fail("E_CONFIG_BINDING", "artifact normalization config hash mismatch")

    inputs = artifact["inputs"]
    if len(inputs) != 2 or {item["role"] for item in inputs} != {
        "source",
        "normalization_config",
    }:
        fail("E_ARTIFACT_BINDING", "artifact inputs must be exactly source and config")
    roles = {item["role"]: item for item in inputs}
    if (
        roles["source"]["path"] != config["origin"]["filename"]
        or roles["source"]["sha256"] != source_hash
    ):
        fail("E_SOURCE_PIN", "artifact source input hash mismatch")
    if (
        roles["normalization_config"]["path"] != f"configs/{config_path.name}"
        or roles["normalization_config"]["sha256"] != config_hash
    ):
        fail("E_CONFIG_BINDING", "artifact config input hash mismatch")
    if artifact["code_version"] != normalizer_bundle_hash():
        fail("E_CODE_BINDING", "normalizer code bundle does not match this checkout")
    current_replay_hash = sha256(read_once(Path(__file__).resolve(), MAX_METADATA_BYTES))
    if current_replay_hash != STARTUP_REPLAY_CODE_SHA256:
        fail("E_CODE_CHANGED", "structural replay source changed during execution")
    gates = artifact.get("gate_results", {})
    for gate in (
        "atomic_run_publish",
        "direct_source_execution",
        "source_snapshot_verified",
    ):
        if gates.get(gate) is not True:
            fail("E_ARTIFACT_BINDING", f"artifact gate is not true: {gate}")
    return {
        "trace_sha256": trace_hash,
        "sidecar_canonical_sha256": sidecar_hash,
        "normalization_config_sha256": config_hash,
        "source_sha256": source_hash,
        "structural_replay_code_sha256": STARTUP_REPLAY_CODE_SHA256,
        "renormalization_verified": False,
    }


def build_result(
    trace_name: str,
    trace_bytes: bytes,
    sidecar_bytes: bytes,
    artifact_bytes: bytes,
    config: dict[str, Any],
    sidecar: dict[str, Any],
    artifact: dict[str, Any],
    records: list[dict[str, Any]],
    service_totals: dict[str, dict[str, int]],
    first_key: tuple[int, int, int],
    last_key: tuple[int, int, int],
    dags: dict[str, tuple[dict[str, Any], str]],
    bindings: dict[str, str],
) -> dict[str, Any]:
    if len(records) != sidecar["output_row_count"]:
        fail("E_ROW_COUNT", "trace row count does not match sidecar")
    services = []
    total_demand = {field: 0 for field in DEMAND_FIELDS}
    for service in sorted(service_totals):
        if service not in dags:
            fail("E_SERVICE_UNKNOWN", f"no static DAG for service: {service}")
        dag, dag_hash = dags[service]
        if dag["provenance"] != sidecar["provenance"]:
            fail("E_DAG_MISMATCH", f"{service}: DAG provenance does not match trace")
        totals = service_totals[service]
        demand = {field: totals[field] for field in DEMAND_FIELDS}
        for field in DEMAND_FIELDS:
            total_demand[field] = checked_add(
                f"total.{field}",
                total_demand[field],
                demand[field],
            )
        services.append(
            {
                "service": service,
                "service_dag_sha256": dag_hash,
                "request_count": totals["request_count"],
                "demand": demand,
                "stages": [
                    {
                        "node_id": node["node_id"],
                        "op_class": node["op_class"],
                        "demand_scope": "covered_request_demand_not_stage_work",
                        "request_count": totals["request_count"],
                        "demand": dict(demand),
                    }
                    for node in dag["nodes"]
                ],
            }
        )
    return {
        "schema_version": 1,
        "kind": "structural_replay",
        "trace": trace_name,
        "record_count": len(records),
        "record_limit": MAX_RECORDS,
        "order_preserved": True,
        "first_order_key": list(first_key),
        "last_order_key": list(last_key),
        "bindings": {
            **bindings,
            "sidecar_file_sha256": sha256(sidecar_bytes),
            "artifact_file_sha256": sha256(artifact_bytes),
            "artifact_run_id": artifact["run_id"],
        },
        "total_demand": total_demand,
        "services": services,
    }


def replay(
    run_dir: Path,
    trace_name: str,
    source_config_path: Path,
    dag_dir: Path,
    source_path: Path,
) -> dict[str, Any]:
    require_jsonschema()
    if not TRACE_NAME_RE.fullmatch(trace_name):
        fail("E_PATH", "trace must be a safe JSONL basename")
    run_dir = run_dir.resolve()
    artifact_path = run_dir / "normalize.artifact.json"
    artifact_bytes = read_once(artifact_path, MAX_METADATA_BYTES)
    config_path = source_config_path.resolve()
    config_bytes = read_once(config_path, MAX_METADATA_BYTES)
    source_bytes = read_once(source_path.resolve(), MAX_TRACE_BYTES)

    artifact = parse_json(artifact_bytes, str(artifact_path))
    if artifact_bytes != canonical_json(artifact) + b"\n":
        fail("E_CANONICAL", "artifact manifest is not canonical JSON")
    config = parse_json(config_bytes, str(config_path))
    validators = {
        "request": load_validator("request.schema.json"),
        "sidecar": load_validator("trace_manifest.schema.json"),
        "artifact": load_validator("artifact_manifest.schema.json"),
        "source_config": load_validator("source_config.schema.json"),
        "dag": load_validator("service_dag.schema.json"),
        "result": load_validator("structural_replay_result.schema.json"),
    }
    validate_schema(validators["artifact"], artifact, "artifact manifest")
    validate_schema(validators["source_config"], config, "source config")
    if artifact["kind"] != "normalize":
        fail("E_ARTIFACT_BINDING", "artifact kind must be normalize")
    dags = load_dags(dag_dir.resolve(), validators["dag"])
    members = validate_artifact_members(
        run_dir,
        artifact,
        config,
        source_bytes,
        validators,
        dags,
    )
    if trace_name not in members:
        fail("E_ARTIFACT_BINDING", "requested trace is not in artifact outputs")
    trace_bytes, sidecar, sidecar_bytes = members[trace_name]
    validate_source_binding(source_bytes, config, sidecar)
    bindings = bind_inputs(
        trace_name,
        trace_bytes,
        sidecar,
        artifact,
        config,
        source_bytes,
        config_path,
    )
    records, totals, first_key, last_key = parse_trace(
        trace_bytes,
        validators["request"],
        config,
        sidecar,
    )
    validate_records_against_window(records, sidecar)
    result = build_result(
        trace_name,
        trace_bytes,
        sidecar_bytes,
        artifact_bytes,
        config,
        sidecar,
        artifact,
        records,
        totals,
        first_key,
        last_key,
        dags,
        bindings,
    )
    validate_schema(validators["result"], result, "structural replay result")
    return result


def verify_renormalization(
    run_dir: Path,
    source_path: Path,
    config_path: Path,
) -> None:
    artifact_path = run_dir.resolve() / "normalize.artifact.json"
    artifact = parse_json(
        read_once(artifact_path, MAX_METADATA_BYTES),
        str(artifact_path),
    )
    outputs = artifact["outputs"]
    scenarios = []
    expected_names = {"normalize.artifact.json"}
    for output in outputs:
        trace_name = output["path"]
        sidecar_name = trace_name.replace(".jsonl", ".manifest.json")
        sidecar = parse_json(
            read_once(run_dir.resolve() / sidecar_name, MAX_METADATA_BYTES),
            sidecar_name,
        )
        scenarios.append(sidecar["window"]["scenario"])
        expected_names.add(trace_name)
        expected_names.add(sidecar_name)
    if len(scenarios) == 1:
        scenario_arg = scenarios[0]
    elif set(scenarios) == set(SCENARIOS) and len(scenarios) == len(SCENARIOS):
        scenario_arg = "all"
    else:
        fail(
            "E_RENORMALIZATION",
            "normalizer supports one scenario or the complete four-scenario set",
        )

    with tempfile.TemporaryDirectory(prefix="s8_structural_replay_") as temporary:
        output_dir = Path(temporary) / "normalized"
        command = [
            sys.executable,
            "-I",
            "-B",
            str(HERE / "normalize_trace.py"),
            "--source-file",
            str(source_path.resolve()),
            "--config",
            str(config_path.resolve()),
            "--scenario",
            scenario_arg,
            "--output-dir",
            str(output_dir),
        ]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            fail("E_RENORMALIZATION", f"cannot rerun pinned normalizer: {exc}")
        if completed.returncode != 0:
            diagnostic = completed.stderr.strip() or completed.stdout.strip()
            fail("E_RENORMALIZATION", f"normalizer failed: {diagnostic}")
        produced = {path.name for path in output_dir.iterdir() if path.is_file()}
        if produced != expected_names:
            fail("E_RENORMALIZATION", "normalizer output file set differs")
        for name in sorted(expected_names):
            original = read_once(run_dir.resolve() / name, MAX_TRACE_BYTES)
            regenerated = read_once(output_dir / name, MAX_TRACE_BYTES)
            if original != regenerated:
                fail("E_RENORMALIZATION", f"regenerated bytes differ: {name}")


# --- semi_synthetic mix (mix-v1) structural replay -------------------------
# The single-source flow above is source-pinned and cannot represent a mix.
# A mix has no single raw source; it binds >=2 real components by hash.

MIX_CODE_FILES = ("compose_mix.py", "normalize_trace.py")
REAL_PROVENANCE = ("real", "real_decomposed")


def mix_bundle_hash() -> str:
    lines = []
    for relpath in sorted(MIX_CODE_FILES):
        digest = hashlib.sha256(read_once(HERE / relpath, MAX_METADATA_BYTES)).hexdigest()
        lines.append(f"{relpath}  {digest}")
    return sha256("\n".join(lines).encode("ascii"))


def validate_mix_sidecar_semantics(
    sidecar: dict[str, Any],
    mix_config: dict[str, Any],
    config_hash: str,
    trace_name: str,
) -> None:
    if sidecar["provenance"] != "semi_synthetic":
        fail("E_MANIFEST_SEMANTICS", "mix sidecar provenance must be semi_synthetic")
    if sidecar["normalization_config_hash"] != config_hash:
        fail("E_CONFIG_BINDING", "mix sidecar normalization config hash mismatch")
    if sidecar["normalizer_version"] != mix_bundle_hash():
        fail("E_CODE_BINDING", "mix code bundle does not match this checkout")
    if trace_name != f"{mix_config['mix_id']}.jsonl":
        fail("E_MANIFEST_SEMANTICS", "mix trace filename does not match mix_id")
    streams = sidecar["streams"]
    ranks = [stream["rank"] for stream in streams]
    if ranks != sorted(ranks) or len(set(ranks)) != len(ranks):
        fail("E_MANIFEST_SEMANTICS", "mix streams must have unique ascending ranks")
    components = {
        (stream["input_output_sha256"], stream["input_manifest_sha256"])
        for stream in streams
    }
    if len(components) != len(streams):
        fail("E_MANIFEST_SEMANTICS", "mix streams bind a component twice")
    config_by_rank = {stream["rank"]: stream for stream in mix_config["streams"]}
    if set(config_by_rank) != set(ranks):
        fail("E_MANIFEST_SEMANTICS", "mix sidecar ranks do not match the mix config")
    for stream in streams:
        committed = config_by_rank[stream["rank"]]
        for key in (
            "source",
            "source_revision",
            "scale_num",
            "scale_den",
            "offset_us",
            "input_output_sha256",
            "input_manifest_sha256",
        ):
            if stream[key] != committed[key]:
                fail("E_MANIFEST_SEMANTICS", f"mix stream rank {stream['rank']}: {key} differs from config")


def verify_mix_components(
    mix_config: dict[str, Any],
    sidecar: dict[str, Any],
    component_run_dirs: dict[int, Path],
) -> list[dict[str, Any]]:
    verified = []
    config_by_rank = {stream["rank"]: stream for stream in mix_config["streams"]}
    for stream in sorted(sidecar["streams"], key=lambda item: item["rank"]):
        rank = stream["rank"]
        if rank not in component_run_dirs:
            fail("E_COMPONENT_MISSING", f"no component run dir for rank {rank}")
        committed = config_by_rank[rank]
        run_dir = component_run_dirs[rank].resolve()
        trace_name = committed["component_trace"]
        if not TRACE_NAME_RE.fullmatch(trace_name):
            fail("E_COMPONENT_FORMAT", f"rank {rank}: unsafe component trace name")
        trace_bytes = read_once(run_dir / trace_name, MAX_TRACE_BYTES)
        sidecar_name = trace_name.replace(".jsonl", ".manifest.json")
        component_sidecar_bytes = read_once(run_dir / sidecar_name, MAX_METADATA_BYTES)
        component_sidecar = parse_json(component_sidecar_bytes, sidecar_name)
        if component_sidecar_bytes != canonical_json(component_sidecar) + b"\n":
            fail("E_CANONICAL", f"rank {rank}: component sidecar is not canonical JSON")
        if component_sidecar["provenance"] not in REAL_PROVENANCE:
            fail("E_COMPONENT_PROVENANCE", f"rank {rank}: component is not a real source")
        if component_sidecar.get("source_revision") != stream["source_revision"]:
            fail("E_COMPONENT_PIN", f"rank {rank}: component source_revision mismatch")
        actual_output = sha256(trace_bytes)
        actual_manifest = sha256(canonical_json(component_sidecar))
        if actual_output != stream["input_output_sha256"]:
            fail("E_COMPONENT_PIN", f"rank {rank}: component trace hash mismatch")
        if actual_manifest != stream["input_manifest_sha256"]:
            fail("E_COMPONENT_PIN", f"rank {rank}: component sidecar hash mismatch")
        if component_sidecar.get("output_sha256") != actual_output:
            fail("E_COMPONENT_PIN", f"rank {rank}: component sidecar output_sha256 mismatch")
        verified.append(
            {
                "rank": rank,
                "source": stream["source"],
                "input_output_sha256": stream["input_output_sha256"],
                "input_manifest_sha256": stream["input_manifest_sha256"],
            }
        )
    return verified


def parse_mix_trace(
    trace_bytes: bytes,
    validator: jsonschema.Draft202012Validator,
    sidecar: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, int]],
    tuple[int, int, int],
    tuple[int, int, int],
    dict[int, int],
]:
    if not trace_bytes:
        fail("E_EMPTY_TRACE", "mixed trace is empty")
    if b"\r" in trace_bytes or not trace_bytes.endswith(b"\n"):
        fail("E_TRACE_FORMAT", "trace must use LF and end with one LF")
    if trace_bytes.endswith(b"\n\n"):
        fail("E_TRACE_FORMAT", "trace has a blank terminal line")
    if trace_bytes.count(b"\n") > MAX_RECORDS:
        fail("E_RECORD_LIMIT", f"trace exceeds {MAX_RECORDS} records")
    stream_source = {stream["rank"]: stream["source"] for stream in sidecar["streams"]}
    stream_counts = {rank: 0 for rank in stream_source}

    raw_lines = trace_bytes[:-1].split(b"\n")
    if not raw_lines or any(line == b"" for line in raw_lines):
        fail("E_EMPTY_TRACE", "trace contains no records or a blank line")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_key: tuple[int, int, int] | None = None
    first_key: tuple[int, int, int] | None = None
    service_totals: dict[str, dict[str, int]] = {}
    for index, line in enumerate(raw_lines):
        if len(line) > MAX_LINE_BYTES:
            fail("E_SIZE_LIMIT", f"trace line {index + 1} exceeds {MAX_LINE_BYTES} bytes")
        record = parse_json(line, f"trace line {index + 1}")
        if canonical_json(record) != line:
            fail("E_TRACE_FORMAT", f"trace line {index + 1} is not canonical JSON")
        validate_schema(validator, record, f"trace line {index + 1}")
        if record["provenance"] != "semi_synthetic":
            fail("E_SOURCE_PIN", f"{record['event_id']}: mixed record must be semi_synthetic")
        key = order_key(record)
        rank = key[1]
        if rank not in stream_source:
            fail("E_EVENT_ID", f"{record['event_id']}: unknown stream rank {rank}")
        if record["source"] != stream_source[rank]:
            fail("E_SOURCE_PIN", f"{record['event_id']}: source does not match stream rank")
        if (
            record["deadline_us"] is not None
            or record["priority_class"] is not None
            or record["deadline_provenance"] != "none"
            or record["priority_provenance"] != "none"
            or record["observed_latency_us"] is not None
        ):
            fail("E_GATE_A_FIELDS", f"{record['event_id']}: non-structural Gate-A fields present")
        event_id = record["event_id"]
        if event_id in seen_ids:
            fail("E_DUPLICATE_EVENT_ID", f"duplicate event_id: {event_id}")
        seen_ids.add(event_id)
        if previous_key is not None and key <= previous_key:
            fail("E_ORDER", f"{event_id}: total-order key {key} <= {previous_key}")
        if first_key is None:
            first_key = key
        previous_key = key
        stream_counts[rank] += 1
        totals = service_totals.setdefault(
            record["service"],
            {"request_count": 0, **{field: 0 for field in DEMAND_FIELDS}},
        )
        totals["request_count"] += 1
        for field in DEMAND_FIELDS:
            totals[field] = checked_add(
                f"{record['service']}.{field}",
                totals[field],
                record[field],
            )
        records.append(record)
    assert first_key is not None and previous_key is not None
    return records, service_totals, first_key, previous_key, stream_counts


def bind_mix_inputs(
    trace_name: str,
    trace_bytes: bytes,
    sidecar: dict[str, Any],
    sidecar_bytes: bytes,
    artifact: dict[str, Any],
    mix_config: dict[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    trace_hash = sha256(trace_bytes)
    sidecar_hash = sha256(canonical_json(sidecar))
    if artifact["kind"] != "normalize":
        fail("E_ARTIFACT_BINDING", "mix artifact kind must be normalize")
    outputs = artifact["outputs"]
    if len(outputs) != 1 or outputs[0]["path"] != trace_name:
        fail("E_ARTIFACT_BINDING", "mix artifact must bind exactly the mixed trace")
    output = outputs[0]
    if output["sha256"] != trace_hash or sidecar["output_sha256"] != trace_hash:
        fail("E_TRACE_HASH", "mixed trace bytes do not match bound output hash")
    if output.get("sidecar_manifest_sha256") != sidecar_hash:
        fail("E_SIDECAR_HASH", "mixed sidecar does not match artifact sidecar hash")
    if artifact["code_version"] != sidecar["normalizer_version"]:
        fail("E_ARTIFACT_BINDING", "mix artifact and sidecar code versions differ")
    if artifact["code_version"] != mix_bundle_hash():
        fail("E_CODE_BINDING", "mix code bundle does not match this checkout")
    if artifact["config_hash"] != config_hash:
        fail("E_CONFIG_BINDING", "mix artifact config hash mismatch")
    if artifact.get("seed") is not None:
        fail("E_ARTIFACT_BINDING", "mix artifact seed must be null")
    if artifact["deterministic_replay_sha256"] != sha256(output["sha256"].encode("ascii")):
        fail("E_ARTIFACT_BINDING", "mix deterministic replay hash mismatch")
    gates = artifact.get("gate_results", {})
    for gate in ("atomic_run_publish", "direct_source_execution", "source_snapshot_verified"):
        if gates.get(gate) is not True:
            fail("E_ARTIFACT_BINDING", f"mix artifact gate is not true: {gate}")
    inputs = artifact["inputs"]
    roles = [item["role"] for item in inputs]
    if roles.count("mix_config") != 1:
        fail("E_ARTIFACT_BINDING", "mix artifact must carry exactly one mix_config input")
    config_input = next(item for item in inputs if item["role"] == "mix_config")
    if config_input["sha256"] != config_hash:
        fail("E_CONFIG_BINDING", "mix artifact config input hash mismatch")
    component_inputs = [item for item in inputs if item["role"] == "component_trace"]
    if len(component_inputs) != len(sidecar["streams"]):
        fail("E_ARTIFACT_BINDING", "mix artifact component input count mismatch")
    component_hashes = {item["sha256"] for item in component_inputs}
    stream_hashes = {stream["input_output_sha256"] for stream in sidecar["streams"]}
    if component_hashes != stream_hashes:
        fail("E_ARTIFACT_BINDING", "mix artifact component inputs do not match streams")
    run_preimage = canonical_json(
        {
            "mix_id": mix_config["mix_id"],
            "config_hash": config_hash,
            "code_version": artifact["code_version"],
            "output_sha256": output["sha256"],
        }
    )
    expected_run_id = "mix-" + hashlib.sha256(run_preimage).hexdigest()[:24]
    if artifact["run_id"] != expected_run_id:
        fail("E_ARTIFACT_BINDING", "mix artifact run_id mismatch")
    current_replay_hash = sha256(read_once(Path(__file__).resolve(), MAX_METADATA_BYTES))
    if current_replay_hash != STARTUP_REPLAY_CODE_SHA256:
        fail("E_CODE_CHANGED", "structural replay source changed during execution")
    return {
        "trace_sha256": trace_hash,
        "sidecar_canonical_sha256": sidecar_hash,
        "mix_config_sha256": config_hash,
        "structural_replay_code_sha256": STARTUP_REPLAY_CODE_SHA256,
        "recompose_verified": False,
        "sidecar_file_sha256": sha256(sidecar_bytes),
        "artifact_file_sha256": sha256(canonical_json(artifact) + b"\n"),
        "artifact_run_id": artifact["run_id"],
    }


def build_mix_result(
    trace_name: str,
    records: list[dict[str, Any]],
    sidecar: dict[str, Any],
    service_totals: dict[str, dict[str, int]],
    first_key: tuple[int, int, int],
    last_key: tuple[int, int, int],
    stream_counts: dict[int, int],
    dags: dict[str, tuple[dict[str, Any], str]],
    bindings: dict[str, Any],
    verified_streams: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(records) != sidecar["output_row_count"]:
        fail("E_ROW_COUNT", "mixed trace row count does not match sidecar")
    services = []
    total_demand = {field: 0 for field in DEMAND_FIELDS}
    for service in sorted(service_totals):
        if service not in dags:
            fail("E_SERVICE_UNKNOWN", f"no static DAG for service: {service}")
        dag, dag_hash = dags[service]
        if dag["provenance"] not in REAL_PROVENANCE:
            fail("E_DAG_MISMATCH", f"{service}: mixed DAG must derive from a real source")
        totals = service_totals[service]
        demand = {field: totals[field] for field in DEMAND_FIELDS}
        for field in DEMAND_FIELDS:
            total_demand[field] = checked_add(
                f"total.{field}",
                total_demand[field],
                demand[field],
            )
        services.append(
            {
                "service": service,
                "service_dag_sha256": dag_hash,
                "request_count": totals["request_count"],
                "demand": demand,
                "stages": [
                    {
                        "node_id": node["node_id"],
                        "op_class": node["op_class"],
                        "demand_scope": "covered_request_demand_not_stage_work",
                        "request_count": totals["request_count"],
                        "demand": dict(demand),
                    }
                    for node in dag["nodes"]
                ],
            }
        )
    stream_bindings = [
        {
            "rank": item["rank"],
            "source": item["source"],
            "record_count": stream_counts[item["rank"]],
            "input_output_sha256": item["input_output_sha256"],
            "input_manifest_sha256": item["input_manifest_sha256"],
        }
        for item in sorted(verified_streams, key=lambda entry: entry["rank"])
    ]
    return {
        "schema_version": 1,
        "kind": "structural_replay_mix",
        "trace": trace_name,
        "record_count": len(records),
        "record_limit": MAX_RECORDS,
        "order_preserved": True,
        "first_order_key": list(first_key),
        "last_order_key": list(last_key),
        "bindings": {**bindings, "streams": stream_bindings},
        "total_demand": total_demand,
        "services": services,
    }


def replay_mix(
    run_dir: Path,
    mix_config_path: Path,
    component_run_dirs: dict[int, Path],
    dag_dir: Path,
) -> dict[str, Any]:
    require_jsonschema()
    run_dir = run_dir.resolve()
    artifact_bytes = read_once(run_dir / "normalize.artifact.json", MAX_METADATA_BYTES)
    artifact = parse_json(artifact_bytes, "mix artifact")
    if artifact_bytes != canonical_json(artifact) + b"\n":
        fail("E_CANONICAL", "mix artifact is not canonical JSON")
    config_bytes = read_once(mix_config_path.resolve(), MAX_METADATA_BYTES)
    mix_config = parse_json(config_bytes, str(mix_config_path))
    config_hash = sha256(canonical_json(mix_config))

    validators = {
        "request": load_validator("request.schema.json"),
        "sidecar": load_validator("trace_manifest.schema.json"),
        "artifact": load_validator("artifact_manifest.schema.json"),
        "mix_config": load_validator("mix_config.schema.json"),
        "dag": load_validator("service_dag.schema.json"),
        "result": load_validator("structural_replay_mix_result.schema.json"),
    }
    validate_schema(validators["artifact"], artifact, "mix artifact")
    validate_schema(validators["mix_config"], mix_config, "mix config")

    outputs = artifact["outputs"]
    if len(outputs) != 1:
        fail("E_ARTIFACT_BINDING", "mix artifact must bind exactly one output")
    trace_name = outputs[0]["path"]
    if not TRACE_NAME_RE.fullmatch(trace_name) or Path(trace_name).name != trace_name:
        fail("E_ARTIFACT_BINDING", f"unsafe mix output path: {trace_name}")
    trace_bytes = read_once(run_dir / trace_name, MAX_TRACE_BYTES)
    sidecar_name = trace_name.replace(".jsonl", ".manifest.json")
    sidecar_bytes = read_once(run_dir / sidecar_name, MAX_METADATA_BYTES)
    sidecar = parse_json(sidecar_bytes, sidecar_name)
    if sidecar_bytes != canonical_json(sidecar) + b"\n":
        fail("E_CANONICAL", "mix sidecar is not canonical JSON")
    validate_schema(validators["sidecar"], sidecar, sidecar_name)
    if sidecar["provenance"] != "semi_synthetic":
        fail("E_MANIFEST_SEMANTICS", "replay_mix requires a semi_synthetic sidecar")

    validate_mix_sidecar_semantics(sidecar, mix_config, config_hash, trace_name)
    verified_streams = verify_mix_components(mix_config, sidecar, component_run_dirs)
    bindings = bind_mix_inputs(
        trace_name, trace_bytes, sidecar, sidecar_bytes, artifact, mix_config, config_hash
    )
    dags = load_dags(dag_dir.resolve(), validators["dag"])
    records, totals, first_key, last_key, stream_counts = parse_mix_trace(
        trace_bytes, validators["request"], sidecar
    )
    if sidecar["output_row_count"] != trace_bytes.count(b"\n"):
        fail("E_ROW_COUNT", "mix sidecar row count mismatch")
    result = build_mix_result(
        trace_name,
        records,
        sidecar,
        totals,
        first_key,
        last_key,
        stream_counts,
        dags,
        bindings,
        verified_streams,
    )
    validate_schema(validators["result"], result, "structural replay mix result")
    return result


def write_atomic(path: Path, data: bytes) -> None:
    if path.exists():
        fail("E_EXISTS", f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = output.name
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except ReplayError:
        raise
    except OSError as exc:
        fail("E_IO", f"cannot publish replay output {path}: {exc}")
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Single-source replay args (required unless --mix-config selects mix mode).
    parser.add_argument("--run-dir")
    parser.add_argument("--trace")
    parser.add_argument("--source-config")
    parser.add_argument("--source")
    parser.add_argument("--dag-dir", default=str(HERE / "service_dags"))
    parser.add_argument("--output")
    # Mixed (semi_synthetic) replay args.
    parser.add_argument("--mix-config", help="mix config (selects mix replay mode)")
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        metavar="RANK:RUN_DIR",
        help="component run directory for a stream rank (repeatable, mix mode)",
    )
    return parser


def _parse_component_args(pairs: list[str]) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for pair in pairs:
        if ":" not in pair:
            fail("E_ARG", f"--component must be rank:run_dir, got {pair}")
        rank_text, path_text = pair.split(":", 1)
        if not CANONICAL_UINT_RE.fullmatch(rank_text):
            fail("E_ARG", f"--component rank must be an integer, got {rank_text}")
        rank = int(rank_text)
        if rank in mapping:
            fail("E_ARG", f"--component rank {rank} specified twice")
        mapping[rank] = Path(path_text)
    return mapping


def _main_mix(args: argparse.Namespace) -> bytes:
    component_run_dirs = _parse_component_args(args.component)
    result = replay_mix(
        Path(args.run_dir),
        Path(args.mix_config).resolve(),
        component_run_dirs,
        Path(args.dag_dir).resolve(),
    )
    return canonical_json(result) + b"\n"


def main(argv: list[str] | None = None) -> int:
    if not DIRECT_SOURCE_EXECUTION:
        print(
            "E_EXECUTION_PROVENANCE: certifying replay requires direct .py CLI execution",
            file=sys.stderr,
        )
        return 1
    args = build_parser().parse_args(argv)
    try:
        if args.mix_config is not None:
            result_bytes = _main_mix(args)
            if args.output:
                write_atomic(Path(args.output).resolve(), result_bytes)
            print(result_bytes.decode("ascii"), end="")
            return 0
        for name in ("run_dir", "trace", "source_config", "source"):
            if getattr(args, name) is None:
                fail("E_ARG", f"single-source replay requires --{name.replace('_', '-')}")
        config_path = Path(args.source_config).resolve()
        dag_path = Path(args.dag_dir).resolve()
        if (
            config_path.parent != (HERE / "configs").resolve()
            or config_path.name
            not in {"burstgpt.config.json", "ragpulse.config.json"}
        ):
            fail("E_PIN_SET", "certifying replay requires a checked-in source config")
        if dag_path != (HERE / "service_dags").resolve():
            fail("E_PIN_SET", "certifying replay requires the checked-in DAG catalog")
        result = replay(
            Path(args.run_dir),
            args.trace,
            config_path,
            dag_path,
            Path(args.source),
        )
        verify_renormalization(
            Path(args.run_dir),
            Path(args.source),
            config_path,
        )
        current_replay_hash = sha256(
            read_once(Path(__file__).resolve(), MAX_METADATA_BYTES)
        )
        if current_replay_hash != STARTUP_REPLAY_CODE_SHA256:
            fail("E_CODE_CHANGED", "structural replay source changed during execution")
        result["bindings"]["renormalization_verified"] = True
        validate_schema(
            load_validator("structural_replay_result.schema.json"),
            result,
            "structural replay result",
        )
        result_bytes = canonical_json(result) + b"\n"
        if args.output:
            write_atomic(Path(args.output).resolve(), result_bytes)
    except ReplayError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"E_INTERNAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(result_bytes.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
