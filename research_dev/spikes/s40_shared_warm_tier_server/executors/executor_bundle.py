#!/usr/bin/env python3
"""Build and validate the immutable S40 Python executor bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
SPIKES = HERE.parents[1]
S39 = SPIKES / "s39_phone_model_switch_trace"

QUALIFICATION_RELATIVES = (
    "CP0_R1_CANDIDATE.json",
    "CP0_R1_EVIDENCE_CONTRACT_V2.json",
    "CP0_R1_EVIDENCE_CONTRACT_V2_1.json",
    "CP0_R1_EVIDENCE_CONTRACT_V2_2.json",
    "CP0_R1_MMLU64_CORPUS_V2_2.jsonl",
    "CP0_R1_MMLU64_SOURCES_V2_2.json",
    "CP0_R1_SHA256SUMS.txt",
    "CP0_R1_TWO_ROUTE_ELIGIBILITY_CONTRACT.json",
    "CP0_R1_V2_SHA256SUMS.txt",
    "CP0_R1_V2_1_SHA256SUMS.txt",
    "CP0_R1_V2_2_SHA256SUMS.txt",
    "RESULTS_CP0_R1.md",
    "RESULTS_CP0_R1_V2.md",
    "RESULTS_CP0_R1_V2_1.md",
    "RESULTS_CP0_R1_V2_2.md",
    "SHARD_MANIFEST.json",
    "build_cp0_r1_contract.py",
    "build_cp0_r1_mmlu64_v22.py",
    "build_cp0_r1_v2.py",
    "build_cp0_r1_v21.py",
    "build_cp0_r1_v22.py",
    "cp0_r1_eligibility.py",
    "cp0_r1_evidence_v2.py",
    "cp0_r1_evidence_v21.py",
    "cp0_r1_evidence_v22.py",
    "cp0_r1_phase_preflight_v21.py",
    "cp0_r1_phase_preflight_v22.py",
    "cp0_r1_preflight_v2.py",
    "tests/test_cp0_r1_eligibility.py",
    "tests/test_cp0_r1_evidence_v2.py",
    "tests/test_cp0_r1_evidence_v21.py",
    "tests/test_cp0_r1_evidence_v22.py",
    "validate_cp0_r1_preflight_v2.py",
    "v23_readiness/CP0_R1_EVIDENCE_CONTRACT_V2_3.json",
    "v23_readiness/build_contract_v23.py",
    "v23_readiness/cp0_r1_evidence_v23.py",
    "v23_readiness/v23_common.py",
)

SOURCES = {
    "a6000_phone_route_control.py": HERE / "a6000_phone_route_control.py",
    "a6000_phone_observer.py": HERE / "a6000_phone_observer.py",
    "a6000_ssh_control.py": HERE / "a6000_ssh_control.py",
    "cache_control.py":
        SPIKES / "s39_desktop_swap_baseline" / "cache_control.py",
    "cache_control_runner.py": HERE / "cache_control_runner.py",
    "desktop_gateway.py": HERE / "desktop_gateway.py",
    "experiment/EXPERIMENT_CONTRACT.json":
        HERE.parent / "EXPERIMENT_CONTRACT.json",
    "executor_bundle.py": HERE / "executor_bundle.py",
    "gateway_bridge.py": HERE / "gateway_bridge.py",
    "phone_gateway.py": HERE / "phone_gateway.py",
    "phone_observer_bridge.py": HERE / "phone_observer_bridge.py",
    "qualification_authority.py": HERE / "qualification_authority.py",
    "readiness_v23.py": HERE / "readiness_v23.py",
    "runtime_binding.py": HERE / "runtime_binding.py",
    "validate_phone_observer.py": HERE / "validate_phone_observer.py",
    "mixed_phase_batcher.py":
        SPIKES / "s39_phone_model_switch_trace" / "mixed_phase_batcher.py",
    "stage_v3_client.py":
        SPIKES / "s22_slo_overlap_pipeline" / "stage_v3_client.py",
    **{
        f"qualification/s39/{relative}": S39 / relative
        for relative in QUALIFICATION_RELATIVES
    },
}

ISOLATED_LAUNCHER = (
    "import runpy,sys;"
    "root=sys.argv[1];script=sys.argv[2];"
    "sys.path.insert(0,root);"
    "sys.argv=[script]+sys.argv[3:];"
    "runpy.run_path(script,run_name='__main__')"
)
LEGACY_PYTHON_FLAGS = ["-B", "-s", "-P"]
ISOLATED_PYTHON_FLAGS = ["-I", "-S", "-B"]


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


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_executor_bundle(output: Path) -> dict[str, Any]:
    require(output.is_absolute() and not output.exists(), "bundle output path")
    output.mkdir(mode=0o700)
    rows = []
    for name, source in sorted(SOURCES.items()):
        require(source.is_absolute() and source.is_file(), f"source {name}")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as inp, target.open("xb", buffering=0) as out:
            shutil.copyfileobj(inp, out, 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        rows.append({
            "bytes": target.stat().st_size,
            "name": name,
            "sha256": digest(target),
            "source_path": str(source),
            "source_sha256": digest(source),
        })
    manifest = {
        "files": rows,
        "python_flags": ISOLATED_PYTHON_FLAGS,
        "schema": "s40-executor-bundle-v2",
    }
    manifest_path = output / "MANIFEST.json"
    write_new(manifest_path, canonical_bytes(manifest))
    fsync_directory(output)
    result = validate_executor_bundle(output, manifest_path)
    return {
        **result,
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "S40_EXECUTOR_BUNDLE": "1",
            "S40_EXECUTOR_BUNDLE_MANIFEST": str(manifest_path),
            "S40_EXECUTOR_BUNDLE_SHA256": digest(manifest_path),
        },
        "python_argv_prefix": [
            sys.executable,
            *ISOLATED_PYTHON_FLAGS,
            "-c",
            ISOLATED_LAUNCHER,
            str(output),
        ],
    }


def validate_executor_bundle(
    bundle: Path,
    manifest_path: Path,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    require(
        bundle.is_absolute()
        and bundle.is_dir()
        and manifest_path == bundle / "MANIFEST.json"
        and manifest_path.is_file(),
        "bundle paths",
    )
    raw = manifest_path.read_bytes()
    if expected_manifest_sha256 is not None:
        require(
            hashlib.sha256(raw).hexdigest() == expected_manifest_sha256,
            "bundle manifest digest",
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"bundle manifest JSON: {error}") from error
    require(canonical_bytes(value) == raw, "bundle manifest canonical bytes")
    require(type(value) is dict
            and set(value) == {"files", "python_flags", "schema"},
            "bundle manifest schema")
    require(
        (
            value["schema"] == "s40-executor-bundle-v1"
            and value["python_flags"] == LEGACY_PYTHON_FLAGS
        )
        or (
            value["schema"] == "s40-executor-bundle-v2"
            and value["python_flags"] == ISOLATED_PYTHON_FLAGS
        ),
        "bundle manifest version",
    )
    expected_names = set(SOURCES)
    rows = value["files"]
    require(type(rows) is list and len(rows) == len(expected_names), "bundle files")
    observed = set()
    for row in rows:
        require(
            type(row) is dict
            and set(row)
            == {"bytes", "name", "sha256", "source_path", "source_sha256"},
            "bundle file row",
        )
        name = row["name"]
        require(
            type(name) is str
            and name in expected_names
            and name not in observed,
            "bundle file name",
        )
        observed.add(name)
        path = bundle / name
        require(
            path.is_file()
            and type(row["bytes"]) is int
            and row["bytes"] > 0
            and type(row["source_path"]) is str
            and Path(row["source_path"]).is_absolute()
            and type(row["source_sha256"]) is str
            and len(row["source_sha256"]) == 64
            and path.stat().st_size == row["bytes"]
            and digest(path) == row["sha256"],
            f"bundle file changed: {name}",
        )
    require(observed == expected_names, "bundle file set")
    actual_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    require(
        actual_files == expected_names | {"MANIFEST.json"},
        "unexpected bundle file",
    )
    require(
        not any(
            path.name == "__pycache__" or path.suffix in (".pyc", ".pyo")
            for path in bundle.rglob("*")
        ),
        "bundle contains bytecode",
    )
    return {
        "bundle_path": str(bundle),
        "files": rows,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "schema": "s40-executor-bundle-validation-v1",
        "status": "PASS",
    }


def validate_runtime_environment(entrypoint: Path) -> None:
    if os.environ.get("S40_EXECUTOR_BUNDLE") != "1":
        return
    manifest_path = Path(
        os.environ.get("S40_EXECUTOR_BUNDLE_MANIFEST", "")
    )
    expected = os.environ.get("S40_EXECUTOR_BUNDLE_SHA256")
    require(
        expected is not None
        and len(expected) == 64
        and entrypoint.parent == manifest_path.parent,
        "executor bundle environment",
    )
    validate_executor_bundle(entrypoint.parent, manifest_path, expected)


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("build", "validate"):
        print(
            "usage: executor_bundle.py build|validate /absolute/bundle",
            file=sys.stderr,
        )
        return 2
    bundle = Path(sys.argv[2])
    result = (
        build_executor_bundle(bundle)
        if sys.argv[1] == "build"
        else validate_executor_bundle(bundle, bundle / "MANIFEST.json")
    )
    print(canonical_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BundleError, OSError) as error:
        print(f"executor bundle failed: {error}", file=sys.stderr)
        raise SystemExit(2)
