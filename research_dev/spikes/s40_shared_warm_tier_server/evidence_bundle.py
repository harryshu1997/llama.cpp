#!/usr/bin/env python3
"""Build and validate the isolated Python acquisition bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
ISOLATED_LAUNCHER = (
    "import runpy,sys;"
    "root=sys.argv[1];script=sys.argv[2];"
    "sys.path.insert(0,root);"
    "sys.argv=[script]+sys.argv[3:];"
    "runpy.run_path(script,run_name='__main__')"
)
LEGACY_PYTHON_FLAGS = ["-B", "-s", "-P"]
ISOLATED_PYTHON_FLAGS = ["-I", "-S", "-B"]
SOURCES = {
    name: HERE / name
    for name in (
        "acquire_trace.py",
        "bridge_overhead.py",
        "campaign_plan.py",
        "campaign_reduce.py",
        "event_evidence.py",
        "evidence_common.py",
        "gpu_isolation.py",
        "physical_orchestrator.py",
        "resource_sampler.py",
        "run_manifest.py",
        "validate_inputs.py",
    )
}


class BundleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def write_new(path: Path, raw: bytes) -> None:
    with path.open("xb", buffering=0) as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def validate_bundle(
        bundle: Path,
        manifest_path: Path,
        expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    require(
        bundle.is_absolute()
        and bundle.is_dir()
        and manifest_path == bundle / "MANIFEST.json"
        and manifest_path.is_file(),
        "evidence bundle paths",
    )
    raw = manifest_path.read_bytes()
    if expected_manifest_sha256 is not None:
        require(
            hashlib.sha256(raw).hexdigest() == expected_manifest_sha256,
            "evidence bundle manifest digest",
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"evidence bundle manifest JSON: {error}") from error
    require(canonical_bytes(value) == raw, "evidence bundle manifest bytes")
    require(type(value) is dict
            and set(value) == {"files", "python_flags", "schema"},
            "evidence bundle manifest schema")
    require(
        (
            value["schema"] == "s40-evidence-bundle-v1"
            and value["python_flags"] == LEGACY_PYTHON_FLAGS
        )
        or (
            value["schema"] == "s40-evidence-bundle-v2"
            and value["python_flags"] == ISOLATED_PYTHON_FLAGS
        ),
        "evidence bundle manifest version",
    )
    rows = value["files"]
    require(type(rows) is list and len(rows) == len(SOURCES),
            "evidence bundle file count")
    observed = set()
    for row in rows:
        require(
            type(row) is dict
            and set(row) == {"bytes", "name", "sha256"},
            "evidence bundle file row",
        )
        name = row["name"]
        require(
            type(name) is str
            and name in SOURCES
            and name not in observed,
            "evidence bundle file name",
        )
        observed.add(name)
        path = bundle / name
        require(
            path.is_file()
            and type(row["bytes"]) is int
            and row["bytes"] > 0
            and path.stat().st_size == row["bytes"]
            and digest(path) == row["sha256"],
            f"evidence bundle file changed: {name}",
        )
    require(observed == set(SOURCES), "evidence bundle file set")
    require(
        {
            path.name for path in bundle.iterdir() if path.is_file()
        } == set(SOURCES) | {"MANIFEST.json"},
        "unexpected evidence bundle file",
    )
    require(
        not any(
            path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
            for path in bundle.rglob("*")
        ),
        "evidence bundle contains bytecode",
    )
    return {
        "files": rows,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "PASS",
    }


def build_bundle(output: Path) -> dict[str, Any]:
    require(output.is_absolute() and not output.exists(),
            "evidence bundle output path")
    output.mkdir(mode=0o700)
    rows = []
    for name, source in sorted(SOURCES.items()):
        require(source.is_absolute() and source.is_file(), f"source {name}")
        target = output / name
        with source.open("rb") as inp, target.open("xb", buffering=0) as out:
            shutil.copyfileobj(inp, out, 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        rows.append({
            "bytes": target.stat().st_size,
            "name": name,
            "sha256": digest(target),
        })
    manifest_path = output / "MANIFEST.json"
    write_new(manifest_path, canonical_bytes({
        "files": rows,
        "python_flags": ISOLATED_PYTHON_FLAGS,
        "schema": "s40-evidence-bundle-v2",
    }))
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    result = validate_bundle(output, manifest_path)
    return {
        **result,
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        "python_argv_prefix": [
            sys.executable,
            *ISOLATED_PYTHON_FLAGS,
            "-c",
            ISOLATED_LAUNCHER,
            str(output),
        ],
    }
