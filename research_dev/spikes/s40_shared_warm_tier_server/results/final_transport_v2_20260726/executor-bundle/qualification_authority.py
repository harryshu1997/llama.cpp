#!/usr/bin/env python3
"""Re-evaluate raw S39 V2.3 qualification evidence for an S40 route."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any

from executor_bundle import validate_executor_bundle
from phone_gateway import (
    MAX_COMMAND_BYTES,
    canonical_bytes,
    exact_keys,
    integer,
    require,
    sha256_text,
    strict_json_loads,
    string,
)


AUTHORITY_SCHEMA = "s40-route-qualification-authority-v1"
DESKTOP_SMOKE_SCHEMA = "s40-desktop-smoke-authority-v1"
DESKTOP_SMOKE_SCOPE = "DESKTOP_SMOKE_ONLY"
EVALUATOR_RELATIVE = (
    "qualification/s39/v23_readiness/cp0_r1_evidence_v23.py"
)
CONTRACT_RELATIVE = (
    "qualification/s39/v23_readiness/"
    "CP0_R1_EVIDENCE_CONTRACT_V2_3.json"
)
CANDIDATE_RELATIVE = "qualification/s39/CP0_R1_CANDIDATE.json"
EXPERIMENT_CONTRACT_RELATIVE = "experiment/EXPERIMENT_CONTRACT.json"
EXPERIMENT_CONTRACT_SHA256 = (
    "1444f5eea4b1e58aed8529c440ffcb89198c090b768688620c363d255feebfda"
)
V22_MANIFEST = "EVIDENCE_BUNDLE.json"
PHASE_SLOT = {"A_ONLY": "A", "B_ONLY": "B"}
PHASE_STATUS = {
    "A_ONLY": "MODEL_A_QUALIFICATION_PASS",
    "B_ONLY": "MODEL_B_QUALIFICATION_PASS",
}
ISOLATED_BOOTSTRAP = (
    "import runpy,sys\n"
    "from pathlib import Path\n"
    "root=Path(sys.argv.pop(1))\n"
    "s39=root/'qualification/s39'\n"
    "v23=s39/'v23_readiness'\n"
    "sys.path[:0]=[str(v23),str(s39)]\n"
    "target=v23/'cp0_r1_evidence_v23.py'\n"
    "sys.argv[0]=str(target)\n"
    "runpy.run_path(str(target),run_name='__main__')\n"
)
FILE_RECORD_KEYS = {"path", "sha256"}
ROOT_RECORD_KEYS = {
    "artifact_snapshot",
    "bundle_manifest_sha256",
    "bundle_root",
    "fresh_snapshot",
    "readiness_lock",
    "runtime_identity",
}


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def absolute_path(value: Any, field: str) -> Path:
    text = string(value, field)
    require(
        text.startswith("/")
        and "\x00" not in text
        and not any(character.isspace() for character in text),
        field,
    )
    return Path(text)


def regular_file(path: Path, field: str) -> None:
    require(
        path.is_file()
        and not path.is_symlink()
        and path.resolve() == path,
        field,
    )


def file_record(value: Any, field: str, root: Path) -> dict[str, Any]:
    value = exact_keys(value, FILE_RECORD_KEYS, field)
    path = absolute_path(value["path"], f"{field}.path")
    regular_file(path, f"{field}.path")
    require(path.is_relative_to(root), f"{field}.root")
    expected = sha256_text(value["sha256"], f"{field}.sha256")
    require(file_sha256(path) == expected, f"{field}.changed")
    return {"path": path, "sha256": expected}


def _manifest_inputs(
    root: Path,
    expected_sha256: str,
) -> tuple[dict[Path, str], dict[str, Any]]:
    manifest = root / V22_MANIFEST
    regular_file(manifest, "qualification bundle manifest")
    raw = manifest.read_bytes()
    require(
        0 < len(raw) <= MAX_COMMAND_BYTES,
        "qualification bundle manifest size",
    )
    require(
        hashlib.sha256(raw).hexdigest() == expected_sha256,
        "qualification bundle manifest changed",
    )
    value = strict_json_loads(raw, "qualification bundle manifest")
    require(
        canonical_bytes(value) == raw,
        "qualification bundle manifest canonical bytes",
    )
    require(type(value) is dict, "qualification bundle manifest type")
    artifacts = value.get("artifacts")
    require(type(artifacts) is list and artifacts, "qualification artifacts")
    inputs = {manifest: expected_sha256}
    for index, artifact in enumerate(artifacts):
        require(type(artifact) is dict, f"qualification artifact[{index}]")
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        require(
            type(relative) is str
            and relative
            and type(expected) is str
            and len(expected) == 64,
            f"qualification artifact[{index}] record",
        )
        pure = PurePosixPath(relative)
        require(
            not pure.is_absolute()
            and ".." not in pure.parts
            and "." not in pure.parts,
            f"qualification artifact[{index}] path",
        )
        path = root.joinpath(*pure.parts)
        regular_file(path, f"qualification artifact[{index}] file")
        require(path not in inputs, "duplicate qualification artifact path")
        require(
            file_sha256(path) == expected,
            f"qualification artifact[{index}] changed",
        )
        inputs[path] = expected
    return inputs, value


def _snapshot_inputs(paths: dict[Path, str]) -> dict[Path, tuple[int, int, str]]:
    result = {}
    for path, expected in sorted(paths.items(), key=lambda item: str(item[0])):
        value = path.stat()
        actual = file_sha256(path)
        require(actual == expected, f"qualification input changed: {path}")
        result[path] = (value.st_ino, value.st_size, actual)
    return result


def _validate_root(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, ROOT_RECORD_KEYS, field)
    root = absolute_path(value["bundle_root"], f"{field}.bundle_root")
    require(
        root.is_dir()
        and not root.is_symlink()
        and root.resolve() == root,
        f"{field}.bundle_root",
    )
    manifest_sha256 = sha256_text(
        value["bundle_manifest_sha256"],
        f"{field}.bundle_manifest_sha256",
    )
    inputs, manifest = _manifest_inputs(root, manifest_sha256)
    records = {
        name: file_record(value[name], f"{field}.{name}", root)
        for name in (
            "artifact_snapshot",
            "readiness_lock",
            "fresh_snapshot",
            "runtime_identity",
        )
    }
    for record in records.values():
        require(record["path"] not in inputs, f"{field}.duplicate_input")
        inputs[record["path"]] = record["sha256"]
    return {
        "bundle_manifest_sha256": manifest_sha256,
        "bundle_root": root,
        "inputs": inputs,
        "manifest": manifest,
        **records,
    }


def _read_candidate(bundle: Path) -> dict[str, str]:
    path = bundle / CANDIDATE_RELATIVE
    regular_file(path, "qualification candidate")
    raw = path.read_bytes()
    require(
        0 < len(raw) <= MAX_COMMAND_BYTES,
        "qualification candidate size",
    )
    value = strict_json_loads(raw, "qualification candidate")
    require(
        canonical_bytes(value) == raw,
        "qualification candidate canonical bytes",
    )
    models = value.get("models") if type(value) is dict else None
    require(type(models) is list and len(models) == 2, "qualification models")
    result = {}
    for index, model in enumerate(models):
        require(type(model) is dict, f"qualification model[{index}]")
        slot = model.get("slot")
        model_id = model.get("model_id")
        require(
            slot in ("A", "B")
            and type(model_id) is str
            and model_id
            and slot not in result,
            f"qualification model[{index}] identity",
        )
        result[slot] = model_id
    require(set(result) == {"A", "B"}, "qualification model slots")
    return result


def _parse_evaluator_output(
    raw: bytes,
    root: dict[str, Any],
    phase: str,
    model_id: str,
) -> dict[str, Any]:
    require(0 < len(raw) <= MAX_COMMAND_BYTES, "V2.3 evaluator output size")
    value = strict_json_loads(raw, "V2.3 evaluator output")
    require(canonical_bytes(value) == raw, "V2.3 evaluator canonical output")
    value = exact_keys(
        value,
        {"derived", "schema", "status"},
        "V2.3 evaluator result",
    )
    require(
        value["schema"] == "s39-cp0-r1-readiness-result-v2.3"
        and value["status"] == "V2_3_READINESS_PASS",
        "V2.3 evaluator status",
    )
    derived = exact_keys(
        value["derived"],
        {
            "artifact_snapshot_sha256",
            "fresh_snapshot_sha256",
            "model_id",
            "phase",
            "phase_id",
            "readiness_lock_sha256",
            "runtime",
            "runtime_identity_sha256",
            "v2_2_bundle_manifest_sha256",
            "v2_2_result_sha256",
        },
        "V2.3 derived result",
    )
    require(derived["phase"] == phase, "V2.3 derived phase")
    require(derived["model_id"] == model_id, "V2.3 derived model")
    string(derived["phase_id"], "V2.3 derived phase ID")
    require(
        derived["v2_2_bundle_manifest_sha256"]
        == root["bundle_manifest_sha256"],
        "V2.3 derived bundle root",
    )
    for result_key, record_key in (
        ("artifact_snapshot_sha256", "artifact_snapshot"),
        ("readiness_lock_sha256", "readiness_lock"),
        ("fresh_snapshot_sha256", "fresh_snapshot"),
        ("runtime_identity_sha256", "runtime_identity"),
    ):
        require(
            derived[result_key] == root[record_key]["sha256"],
            f"V2.3 derived {record_key}",
        )
    sha256_text(derived["v2_2_result_sha256"], "V2.2 derived result")
    require(type(derived["runtime"]) is dict, "V2.3 derived runtime")
    return derived


def _read_phase_lock_sha256(
    root: dict[str, Any],
    phase: str,
    phase_id: str,
) -> str:
    path = root["readiness_lock"]["path"]
    raw = path.read_bytes()
    value = strict_json_loads(raw, "qualified readiness lock")
    require(canonical_bytes(value) == raw, "qualified readiness lock canonical")
    value = exact_keys(
        value,
        {
            "artifact_snapshot_sha256",
            "event_ns",
            "phase",
            "phase_id",
            "schema",
            "v2_2_phase_lock_sha256",
        },
        "qualified readiness lock",
    )
    require(
        value["schema"] == "s39-cp0-r1-readiness-lock-v2.3"
        and value["phase"] == phase
        and value["phase_id"] == phase_id,
        "qualified readiness lock identity",
    )
    return sha256_text(
        value["v2_2_phase_lock_sha256"],
        "qualified V2.2 phase lock SHA-256",
    )


def _run_evaluator(
    executor_bundle: Path,
    root: dict[str, Any],
    phase: str,
    model_id: str,
    timeout_seconds: int,
    a_root: dict[str, Any] | None,
) -> dict[str, Any]:
    evaluator = executor_bundle / EVALUATOR_RELATIVE
    contract = executor_bundle / CONTRACT_RELATIVE
    candidate = executor_bundle / CANDIDATE_RELATIVE
    for path, field in (
        (evaluator, "V2.3 evaluator"),
        (contract, "V2.3 contract"),
        (candidate, "V2.3 candidate"),
    ):
        regular_file(path, field)
    before = _snapshot_inputs(root["inputs"])
    a_before = None
    if a_root is not None:
        a_before = _snapshot_inputs(a_root["inputs"])
    command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        ISOLATED_BOOTSTRAP,
        str(executor_bundle),
        "--contract",
        str(contract),
        "--candidate",
        str(candidate),
        "--bundle-root",
        str(root["bundle_root"]),
        "--artifact-snapshot",
        str(root["artifact_snapshot"]["path"]),
        "--readiness-lock",
        str(root["readiness_lock"]["path"]),
        "--fresh-snapshot",
        str(root["fresh_snapshot"]["path"]),
        "--runtime-identity",
        str(root["runtime_identity"]["path"]),
    ]
    if a_root is not None:
        command.extend(["--a-bundle-root", str(a_root["bundle_root"])])
    completed = subprocess.run(
        command,
        cwd="/",
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    require(completed.returncode == 0, "V2.3 evaluator refused evidence")
    require(completed.stderr == b"", "V2.3 evaluator stderr")
    require(
        _snapshot_inputs(root["inputs"]) == before,
        "qualification inputs changed during evaluation",
    )
    if a_root is not None:
        require(
            _snapshot_inputs(a_root["inputs"]) == a_before,
            "qualification A chain changed during evaluation",
        )
    return _parse_evaluator_output(completed.stdout, root, phase, model_id)


def validate_route_authority(
    value: Any,
    *,
    expected_model_id: str,
    expected_phase: str,
    expected_slot: str,
    expected_artifact_certificate_sha256: str,
    expected_readiness_lock_sha256: str,
    executor_bundle: Path,
    executor_bundle_manifest_sha256: str,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {"a_chain", "current", "model_id", "phase", "schema", "slot"},
        "route qualification authority",
    )
    require(value["schema"] == AUTHORITY_SCHEMA, "route authority schema")
    require(expected_phase in PHASE_SLOT, "route authority expected phase")
    require(value["phase"] == expected_phase, "route authority phase")
    require(value["slot"] == expected_slot, "route authority slot")
    require(PHASE_SLOT[expected_phase] == expected_slot, "route phase slot")
    require(value["model_id"] == expected_model_id, "route authority model")
    timeout_seconds = integer(
        timeout_seconds,
        "route authority timeout",
        1,
    )
    require(timeout_seconds <= 300, "route authority timeout")
    executor_bundle = executor_bundle.resolve()
    validate_executor_bundle(
        executor_bundle,
        executor_bundle / "MANIFEST.json",
        sha256_text(
            executor_bundle_manifest_sha256,
            "executor bundle manifest SHA-256",
        ),
    )
    models = _read_candidate(executor_bundle)
    require(
        models[expected_slot] == expected_model_id,
        "route authority candidate model",
    )
    current = _validate_root(value["current"], "route authority current")
    require(
        current["artifact_snapshot"]["sha256"]
        == sha256_text(
            expected_artifact_certificate_sha256,
            "expected artifact certificate SHA-256",
        ),
        "route authority artifact certificate",
    )
    require(
        current["readiness_lock"]["sha256"]
        == sha256_text(
            expected_readiness_lock_sha256,
            "expected readiness lock SHA-256",
        ),
        "route authority readiness lock",
    )

    a_derived = None
    a_root = None
    if expected_phase == "A_ONLY":
        require(value["a_chain"] is None, "A_ONLY authority has an A chain")
    else:
        a_root = _validate_root(value["a_chain"], "route authority A chain")
        require(
            a_root["bundle_root"] != current["bundle_root"],
            "route authority reused A bundle",
        )
        a_derived = _run_evaluator(
            executor_bundle,
            a_root,
            "A_ONLY",
            models["A"],
            timeout_seconds,
            None,
        )
    derived = _run_evaluator(
        executor_bundle,
        current,
        expected_phase,
        expected_model_id,
        timeout_seconds,
        a_root,
    )
    if a_derived is not None:
        require(
            a_derived["phase_id"] != derived["phase_id"],
            "route authority reused phase ID",
        )
    validate_executor_bundle(
        executor_bundle,
        executor_bundle / "MANIFEST.json",
        executor_bundle_manifest_sha256,
    )
    return {
        "a_chain_phase_id":
            None if a_derived is None else a_derived["phase_id"],
        "bundle_manifest_sha256": current["bundle_manifest_sha256"],
        "model_id": expected_model_id,
        "phase": expected_phase,
        "phase_id": derived["phase_id"],
        "phase_lock_sha256": _read_phase_lock_sha256(
            current,
            expected_phase,
            derived["phase_id"],
        ),
        "schema": "s40-route-qualification-derived-v1",
        "scope": "QUALIFIED_ROUTE",
        "status": PHASE_STATUS[expected_phase],
        "v2_2_result_sha256": derived["v2_2_result_sha256"],
    }


def validate_desktop_smoke_authority(
    value: Any,
    *,
    expected_model_id: str,
    expected_model_path: str,
    executor_bundle: Path,
    executor_bundle_manifest_sha256: str,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "gpu_uuid",
            "model_bytes",
            "model_id",
            "model_path",
            "model_sha256",
            "phase",
            "phase_id",
            "schema",
            "scope",
        },
        "desktop smoke authority",
    )
    require(value["schema"] == DESKTOP_SMOKE_SCHEMA, "desktop smoke schema")
    require(value["scope"] == DESKTOP_SMOKE_SCOPE, "desktop smoke scope")
    require(value["phase"] == "DESKTOP_SMOKE", "desktop smoke phase")
    string(value["phase_id"], "desktop smoke phase ID")
    require(value["model_id"] == expected_model_id, "desktop smoke model")
    require(value["model_path"] == expected_model_path, "desktop smoke path")
    executor_bundle = executor_bundle.resolve()
    validate_executor_bundle(
        executor_bundle,
        executor_bundle / "MANIFEST.json",
        executor_bundle_manifest_sha256,
    )
    contract_path = executor_bundle / EXPERIMENT_CONTRACT_RELATIVE
    regular_file(contract_path, "desktop smoke experiment contract")
    contract_raw = contract_path.read_bytes()
    require(
        hashlib.sha256(contract_raw).hexdigest() == EXPERIMENT_CONTRACT_SHA256,
        "desktop smoke experiment contract changed",
    )
    contract = strict_json_loads(contract_raw, "desktop smoke experiment contract")
    require(
        canonical_bytes(contract) == contract_raw,
        "desktop smoke experiment contract canonical bytes",
    )
    require(
        type(contract) is dict
        and contract.get("schema") == "s40-shared-warm-tier-experiment-v1",
        "desktop smoke experiment contract schema",
    )
    runtime = contract.get("runtime_source")
    require(type(runtime) is dict, "desktop smoke runtime source")
    models = runtime.get("models")
    require(
        type(models) is dict and expected_model_id in models,
        "desktop smoke model is outside experiment contract",
    )
    frozen_model = models[expected_model_id]
    require(
        type(frozen_model) is dict
        and set(frozen_model) == {"bytes", "sha256"},
        "desktop smoke frozen model",
    )
    require(
        value["gpu_uuid"] == runtime.get("expected_gpu_uuid"),
        "desktop smoke GPU",
    )
    path = absolute_path(value["model_path"], "desktop smoke model path")
    regular_file(path, "desktop smoke model file")
    expected = sha256_text(
        frozen_model["sha256"],
        "desktop smoke frozen model SHA-256",
    )
    require(value["model_sha256"] == expected, "desktop smoke model digest")
    require(
        integer(value["model_bytes"], "desktop smoke model bytes", 1)
        == integer(frozen_model["bytes"], "desktop smoke frozen model bytes", 1)
        == path.stat().st_size,
        "desktop smoke model size",
    )
    require(file_sha256(path) == expected, "desktop smoke model changed")
    return {
        "model_id": expected_model_id,
        "phase": "DESKTOP_SMOKE",
        "phase_id": value["phase_id"],
        "schema": "s40-desktop-smoke-derived-v1",
        "scope": DESKTOP_SMOKE_SCOPE,
        "status": "DESKTOP_B1_B8_SMOKE_AUTHORIZED",
    }
