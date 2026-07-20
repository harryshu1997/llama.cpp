#!/usr/bin/python3
"""Deterministic mix-v1 composition for S8 (semi_synthetic).

Superposes already-normalized real component traces into one merged
`semi_synthetic` trace, following the frozen NORMALIZATION_SPEC section 7 rules:

  t_mix = floor(t_component * scale_num / scale_den) + offset_us      (checked int)
  merged order key = (t_mix, rank, source_row_id)
  provenance rewritten to semi_synthetic; event_id -> mix:<rank>:<source>:<row>

This module reuses the low-level helpers of normalize_trace.py (canonical JSON,
hashing, record validation) so the frozen single-source normalizer bytes are not
touched. Its certifying entrypoint is direct source execution only.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import normalize_trace as nt  # noqa: E402

# The mix code bundle binds this file plus the reused normalizer helpers.
MIX_CODE_FILES = ("compose_mix.py", "normalize_trace.py")
SAFE_MAX = 9007199254740991
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
TRACE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.jsonl$")
CANONICAL_UINT_RE = re.compile(r"^(0|[1-9][0-9]*)$")
REAL_PROVENANCE = ("real", "real_decomposed")


class MixError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def fail(code: str, message: str) -> None:
    raise MixError(code, message)


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


def canonical_json(value: Any) -> bytes:
    return nt.canonical_json(value)


def sha256_bytes(data: bytes) -> str:
    return nt.sha256_bytes(data)


def mix_bundle_hash() -> str:
    lines = []
    for relpath in sorted(MIX_CODE_FILES):
        digest = nt.sha256_file(HERE / relpath).removeprefix("sha256:")
        lines.append(f"{relpath}  {digest}")
    return sha256_bytes("\n".join(lines).encode("ascii"))


STARTUP_MIX_VERSION = (
    mix_bundle_hash()
    if DIRECT_SOURCE_EXECUTION or __name__ != "__main__"
    else "sha256:" + "0" * 64
)


def checked_mul(name: str, left: int, right: int) -> int:
    product = left * right
    if product > SAFE_MAX:
        fail("E_OVERFLOW", f"{name}: integer overflow ({left} * {right})")
    return product


def checked_add(name: str, left: int, right: int) -> int:
    total = left + right
    if total > SAFE_MAX:
        fail("E_OVERFLOW", f"{name}: integer overflow ({left} + {right})")
    return total


def _load_json_bytes(data: bytes, label: str) -> Any:
    try:
        return nt.load_json_strict_bytes(data, label)
    except nt.NormalizeError as exc:
        fail(exc.code, exc.message)


def _read_file(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        fail("E_IO", f"cannot read {label}: {exc}")
    return b""  # unreachable


def validate_mix_config(config: Any) -> dict[str, Any]:
    """Structural fail-closed validation of a mix config (no jsonschema)."""
    if not isinstance(config, dict):
        fail("E_CONFIG", "mix config must be a JSON object")
    allowed = {"schema_version", "provenance", "mix_id", "streams"}
    extra = set(config) - allowed
    if extra:
        fail("E_CONFIG", f"mix config has unexpected keys: {sorted(extra)}")
    if config.get("schema_version") != 1:
        fail("E_CONFIG", "mix config schema_version must be 1")
    if config.get("provenance") != "semi_synthetic":
        fail("E_CONFIG", "mix config provenance must be semi_synthetic")
    mix_id = config.get("mix_id")
    if not isinstance(mix_id, str) or not SAFE_NAME_RE.fullmatch(mix_id):
        fail("E_CONFIG", "mix config mix_id must be a safe name")
    streams = config.get("streams")
    if not isinstance(streams, list) or len(streams) < 2:
        fail("E_CONFIG", "mix config requires at least two streams")
    seen_ranks: set[int] = set()
    seen_components: set[tuple[str, str]] = set()
    previous_rank: int | None = None
    stream_keys = {
        "rank",
        "source",
        "source_revision",
        "component_trace",
        "scale_num",
        "scale_den",
        "offset_us",
        "input_output_sha256",
        "input_manifest_sha256",
    }
    for index, stream in enumerate(streams):
        if not isinstance(stream, dict) or set(stream) != stream_keys:
            fail("E_CONFIG", f"stream {index}: keys must be exactly {sorted(stream_keys)}")
        rank = stream["rank"]
        if type(rank) is not int or not 0 <= rank <= SAFE_MAX:
            fail("E_CONFIG", f"stream {index}: rank must be a nonnegative integer")
        if rank in seen_ranks:
            fail("E_CONFIG", f"stream {index}: duplicate rank {rank}")
        seen_ranks.add(rank)
        if previous_rank is not None and rank <= previous_rank:
            fail("E_CONFIG", "streams must be listed in strictly ascending rank order")
        previous_rank = rank
        for key in ("source", "source_revision"):
            if not isinstance(stream[key], str) or not stream[key]:
                fail("E_CONFIG", f"stream {index}: {key} must be a non-empty string")
        trace = stream["component_trace"]
        if not isinstance(trace, str) or not TRACE_NAME_RE.fullmatch(trace):
            fail("E_CONFIG", f"stream {index}: component_trace must be a safe jsonl name")
        for key in ("scale_num", "scale_den"):
            if type(stream[key]) is not int or stream[key] < 1:
                fail("E_CONFIG", f"stream {index}: {key} must be a positive integer")
        if type(stream["offset_us"]) is not int or not 0 <= stream["offset_us"] <= SAFE_MAX:
            fail("E_CONFIG", f"stream {index}: offset_us must be a nonnegative integer")
        for key in ("input_output_sha256", "input_manifest_sha256"):
            if not isinstance(stream[key], str) or not SHA256_RE.fullmatch(stream[key]):
                fail("E_CONFIG", f"stream {index}: {key} must be sha256:<64hex>")
        component = (stream["input_output_sha256"], stream["input_manifest_sha256"])
        if component in seen_components:
            fail("E_CONFIG", f"stream {index}: duplicate component binding")
        seen_components.add(component)
    return config


def _parse_component_records(
    trace_bytes: bytes,
    sidecar: dict[str, Any],
    stream: dict[str, Any],
) -> list[tuple[dict[str, Any], int]]:
    if not trace_bytes or trace_bytes[-1:] != b"\n" or b"\r" in trace_bytes:
        fail("E_COMPONENT_FORMAT", f"{stream['source']}: trace must use LF and end with LF")
    if trace_bytes.endswith(b"\n\n"):
        fail("E_COMPONENT_FORMAT", f"{stream['source']}: trace has a blank terminal line")
    provenance = sidecar["provenance"]
    if provenance not in REAL_PROVENANCE:
        fail("E_COMPONENT_PROVENANCE", f"{stream['source']}: component must be real")
    if sidecar.get("source_revision") != stream["source_revision"]:
        fail("E_COMPONENT_PIN", f"{stream['source']}: source_revision differs from config")
    records: list[tuple[dict[str, Any], int]] = []
    prefix = stream["source"] + ":"
    for index, line in enumerate(trace_bytes[:-1].split(b"\n")):
        if line == b"":
            fail("E_COMPONENT_FORMAT", f"{stream['source']}: blank component line")
        record = _load_json_bytes(line, f"{stream['source']} line {index + 1}")
        if canonical_json(record) != line:
            fail("E_COMPONENT_FORMAT", f"{stream['source']} line {index + 1}: not canonical")
        if record.get("provenance") != provenance:
            fail("E_COMPONENT_PROVENANCE", f"{stream['source']} line {index + 1}: provenance")
        if record.get("source") != stream["source"]:
            fail("E_COMPONENT_SOURCE", f"{stream['source']} line {index + 1}: source mismatch")
        event_id = record.get("event_id")
        if not isinstance(event_id, str) or not event_id.startswith(prefix):
            fail("E_COMPONENT_EVENT_ID", f"{stream['source']} line {index + 1}: event_id")
        row_text = event_id[len(prefix):]
        if not CANONICAL_UINT_RE.fullmatch(row_text):
            fail("E_COMPONENT_EVENT_ID", f"{stream['source']} line {index + 1}: row id")
        records.append((record, int(row_text)))
    if not records:
        fail("E_COMPONENT_EMPTY", f"{stream['source']}: component trace is empty")
    return records


def _map_event(
    record: dict[str, Any],
    row_id: int,
    stream: dict[str, Any],
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    t_component = record["t_us"]
    if type(t_component) is not int or t_component < 0:
        fail("E_COMPONENT_TIME", f"{stream['source']}:{row_id}: t_us must be nonnegative int")
    product = checked_mul(f"{stream['source']}:{row_id} scale", t_component, stream["scale_num"])
    quotient = product // stream["scale_den"]
    t_mix = checked_add(f"{stream['source']}:{row_id} offset", quotient, stream["offset_us"])
    mapped = dict(record)
    mapped["event_id"] = f"mix:{stream['rank']}:{stream['source']}:{row_id}"
    mapped["provenance"] = "semi_synthetic"
    mapped["t_us"] = t_mix
    try:
        nt.validate_record(mapped)
    except nt.NormalizeError as exc:
        fail(exc.code, exc.message)
    return mapped, (t_mix, stream["rank"], row_id)


def build_mix_sidecar(
    config: dict[str, Any],
    streams: list[dict[str, Any]],
    output_row_count: int,
    output_sha256: str,
    code_version: str,
    config_hash: str,
) -> dict[str, Any]:
    stream_records = [
        {
            "source": stream["source"],
            "source_revision": stream["source_revision"],
            "rank": stream["rank"],
            "scale_num": stream["scale_num"],
            "scale_den": stream["scale_den"],
            "offset_us": stream["offset_us"],
            "input_output_sha256": stream["input_output_sha256"],
            "input_manifest_sha256": stream["input_manifest_sha256"],
        }
        for stream in sorted(streams, key=lambda item: item["rank"])
    ]
    return {
        "schema_version": 1,
        "provenance": "semi_synthetic",
        "normalizer_version": code_version,
        "normalization_config_hash": config_hash,
        "output_row_count": output_row_count,
        "output_sha256": output_sha256,
        "streams": stream_records,
    }


def build_mix_artifact(
    config: dict[str, Any],
    config_path: Path,
    streams: list[dict[str, Any]],
    output_name: str,
    output_sha256: str,
    sidecar_sha256: str,
    code_version: str,
    config_hash: str,
) -> dict[str, Any]:
    inputs = [
        {
            "role": "mix_config",
            "path": f"configs/{config_path.name}",
            "sha256": config_hash,
        }
    ]
    for stream in sorted(streams, key=lambda item: item["rank"]):
        inputs.append(
            {
                "role": "component_trace",
                "path": f"component/rank{stream['rank']}/{stream['component_trace']}",
                "sha256": stream["input_output_sha256"],
            }
        )
    replay_preimage = output_sha256.encode("ascii")
    run_preimage = canonical_json(
        {
            "mix_id": config["mix_id"],
            "config_hash": config_hash,
            "code_version": code_version,
            "output_sha256": output_sha256,
        }
    )
    run_id = "mix-" + nt.hashlib.sha256(run_preimage).hexdigest()[:24]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "kind": "normalize",
        "inputs": inputs,
        "code_version": code_version,
        "config_hash": config_hash,
        "seed": None,
        "outputs": [
            {
                "path": output_name,
                "sha256": output_sha256,
                "sidecar_manifest_sha256": sidecar_sha256,
            }
        ],
        "deterministic_replay_sha256": sha256_bytes(replay_preimage),
        "gate_results": {
            "atomic_run_publish": True,
            "direct_source_execution": DIRECT_SOURCE_EXECUTION,
            "source_snapshot_verified": True,
            "mix_composition": True,
        },
    }


def _publish(output_dir: Path, files: dict[str, bytes], code_version: str) -> None:
    if output_dir.exists():
        fail("E_EXISTS", f"output directory already exists: {output_dir}")
    parent = output_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".tmp", dir=parent)
        )
    except OSError as exc:
        fail("E_IO", f"cannot create staging directory: {exc}")
    published = False
    try:
        for name, data in files.items():
            nt._write_bytes_fsync(staging / name, data)
        nt._fsync_directory(staging)
        if mix_bundle_hash() != code_version:
            fail("E_CODE_CHANGED", "mix source changed during the run")
        if output_dir.exists():
            fail("E_EXISTS", f"output directory appeared during run: {output_dir}")
        os.rename(staging, output_dir)
        published = True
        nt._fsync_directory(parent)
    except MixError:
        raise
    except OSError as exc:
        fail("E_IO", f"cannot publish run {output_dir}: {exc}")
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


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
        mapping[rank] = Path(path_text).resolve()
    return mapping


def compose(args: argparse.Namespace) -> dict[str, Any]:
    if not DIRECT_SOURCE_EXECUTION:
        fail("E_EXECUTION_PROVENANCE", "certified composition requires direct CLI source execution")
    return _compose_impl(args)


def _compose_impl(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    code_version = STARTUP_MIX_VERSION
    config = validate_mix_config(nt.load_json_strict(config_path))
    config_hash = sha256_bytes(canonical_json(config))
    components = _parse_component_args(args.component)

    merged: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for stream in sorted(config["streams"], key=lambda item: item["rank"]):
        rank = stream["rank"]
        if rank not in components:
            fail("E_ARG", f"no --component run_dir provided for rank {rank}")
        run_dir = components[rank]
        trace_path = run_dir / stream["component_trace"]
        sidecar_path = run_dir / stream["component_trace"].replace(".jsonl", ".manifest.json")
        trace_bytes = _read_file(trace_path, f"component trace {trace_path}")
        sidecar_bytes = _read_file(sidecar_path, f"component sidecar {sidecar_path}")
        sidecar = _load_json_bytes(sidecar_bytes, str(sidecar_path))
        if sidecar_bytes != canonical_json(sidecar) + b"\n":
            fail("E_CANONICAL", f"{sidecar_path}: component sidecar is not canonical JSON")
        actual_output = sha256_bytes(trace_bytes)
        actual_manifest = sha256_bytes(canonical_json(sidecar))
        if actual_output != stream["input_output_sha256"]:
            fail("E_COMPONENT_PIN", f"{stream['source']}: component trace hash mismatch")
        if actual_manifest != stream["input_manifest_sha256"]:
            fail("E_COMPONENT_PIN", f"{stream['source']}: component sidecar hash mismatch")
        if sidecar.get("output_sha256") != actual_output:
            fail("E_COMPONENT_PIN", f"{stream['source']}: sidecar output_sha256 mismatch")
        records = _parse_component_records(trace_bytes, sidecar, stream)
        for record, row_id in records:
            mapped, key = _map_event(record, row_id, stream)
            merged.append((key, mapped))

    merged.sort(key=lambda item: item[0])
    seen_ids: set[str] = set()
    ordered_records = []
    previous_key: tuple[int, int, int] | None = None
    for key, record in merged:
        if previous_key is not None and key <= previous_key:
            fail("E_ORDER", f"non-total mixed order at key {key}")
        previous_key = key
        event_id = record["event_id"]
        if event_id in seen_ids:
            fail("E_DUPLICATE_EVENT_ID", f"duplicate mixed event_id: {event_id}")
        seen_ids.add(event_id)
        ordered_records.append(record)

    output_bytes = b"".join(canonical_json(record) + b"\n" for record in ordered_records)
    if not output_bytes:
        fail("E_EMPTY_OUTPUT", "mixed output is empty")
    output_sha = sha256_bytes(output_bytes)
    output_name = f"{config['mix_id']}.jsonl"
    manifest_name = f"{config['mix_id']}.manifest.json"

    sidecar = build_mix_sidecar(
        config, config["streams"], len(ordered_records), output_sha, code_version, config_hash
    )
    sidecar_bytes = canonical_json(sidecar)
    sidecar_sha = sha256_bytes(sidecar_bytes)
    artifact = build_mix_artifact(
        config,
        config_path,
        config["streams"],
        output_name,
        output_sha,
        sidecar_sha,
        code_version,
        config_hash,
    )
    artifact_bytes = canonical_json(artifact)

    output_dir = Path(args.output_dir).resolve()
    _publish(
        output_dir,
        {
            output_name: output_bytes,
            manifest_name: sidecar_bytes + b"\n",
            "normalize.artifact.json": artifact_bytes + b"\n",
        },
        code_version,
    )
    return {
        "status": "composed",
        "mix_id": config["mix_id"],
        "output_dir": str(output_dir),
        "output_row_count": len(ordered_records),
        "output_sha256": output_sha,
        "sidecar_manifest_sha256": sidecar_sha,
        "artifact_run_id": artifact["run_id"],
        "deterministic_replay_sha256": artifact["deterministic_replay_sha256"],
        "streams": [
            {
                "rank": stream["rank"],
                "source": stream["source"],
                "input_output_sha256": stream["input_output_sha256"],
            }
            for stream in sorted(config["streams"], key=lambda item: item["rank"])
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="mix config JSON")
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        metavar="RANK:RUN_DIR",
        help="component run directory for a stream rank (repeatable)",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compose(args)
    except MixError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"E_INTERNAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(canonical_json(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
