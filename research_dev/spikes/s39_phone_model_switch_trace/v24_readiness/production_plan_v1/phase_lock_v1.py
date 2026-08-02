#!/usr/bin/env python3
"""Materialize the V2.4 A_ONLY phase lock from captured preparation."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
import types
from typing import Callable


def _load_source(name: str, path: Path):
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"E_SOURCE_REGULAR: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise RuntimeError(f"E_SOURCE_CHANGED: {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(bytes(raw), str(path), "exec"), module.__dict__)
    return module


common = _load_source(
    "s39_v24_production_common",
    Path(__file__).resolve().with_name("production_common_v1.py"),
)
CONFIRMATION = "RUN_V24_PHASE_LOCK_PREFLIGHT_A_ONLY"


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(common.canonical_bytes(row) for row in rows)


def _phase_row(role: str, kind: str, phase_id: str, event_ns: int) -> dict:
    return {
        "acquisition_id": phase_id,
        "event_ns": event_ns,
        "kind": kind,
        "phase": common.PHASE,
        "phase_id": phase_id,
        "role": role,
    }


def _digest_json(value) -> str:
    return hashlib.sha256(common.canonical_bytes(value)).hexdigest()


def _preflight_raw(
    preflight_capture: dict,
    phase_id: str,
) -> bytes:
    rows = []
    probes = preflight_capture["probes"]
    for label in sorted(probes):
        value = probes[label]
        rows.append(
            {
                **_phase_row(
                    "phase.preflight",
                    "probe",
                    phase_id,
                    value["completed_ns"],
                ),
                "argv": value["argv"],
                "label": label,
                "returncode": value["returncode"],
                "started_ns": value["started_ns"],
                "stderr": value["stderr"],
                "stdout": value["stdout"],
                "timed_out": False,
            }
        )
    completed = max(value["completed_ns"] for value in probes.values())
    rows.append(
        {
            **_phase_row("phase.preflight", "meta", phase_id, completed),
            "completed_ns": completed,
            "forbidden_work_executed": False,
            "probe_labels": sorted(probes),
        }
    )
    return _jsonl(rows)


def _corpus_raw(corpus_raw: bytes, phase_id: str, event_ns: int) -> bytes:
    rows = []
    for index, line in enumerate(corpus_raw.splitlines(keepends=True)):
        item = common.parse_json(line, f"quality_corpus[{index}]")
        rows.append(
            {
                **_phase_row(
                    "quality.corpus",
                    "item",
                    phase_id,
                    event_ns + index,
                ),
                **item,
            }
        )
    common.require(len(rows) == 64, "E_CORPUS_ITEMS")
    return _jsonl(rows)


def _route_lock_raw(
    contract: dict,
    candidate: dict,
    phase_id: str,
    event_ns: int,
) -> bytes:
    role = f"model.{common.MODEL_ID}.route_lock"
    model = next(
        value
        for value in candidate["models"]
        if value["slot"] == "A" and value["model_id"] == common.MODEL_ID
    )
    route = contract["incumbent_route_lock"]
    geometry = contract["model_geometry"][common.MODEL_ID]
    row = {
        **_phase_row(role, "route_lock", phase_id, event_ns),
        "activation_dtype": geometry["activation_dtype"],
        "activation_element_bytes": geometry["activation_element_bytes"],
        "backend": route["backend"],
        "batch_config_sha256": _digest_json(contract["serving_envelope"]),
        "clock_id": contract["phase_protocol"]["clock_id"],
        "cuda_model_path": geometry["cuda_model_path"],
        "cut_layer": route["cut_layer"],
        "frozen_ns": event_ns,
        "hidden_size": geometry["hidden_size"],
        "model_id": model["model_id"],
        "model_sha256": model["artifact"]["sha256"],
        "n_layer": model["n_layer"],
    }
    for phone in ("op12", "op15"):
        shard = geometry["known_shards"][phone]
        row[f"{phone}_shard_bytes"] = shard["bytes"]
        row[f"{phone}_shard_path"] = shard["path"]
        row[f"{phone}_shard_sha256"] = shard["sha256"]
        row[f"{phone}_stored_layers"] = route[f"{phone}_stored_layers"]
    return _jsonl([row])


def _phase_lock_raw(
    contract: dict,
    candidate_raw: bytes,
    phase_id: str,
    event_ns: int,
    corpus_raw: bytes,
    route_raw: bytes,
) -> bytes:
    role = "phase.lock"
    row = {
        **_phase_row(role, "phase_lock", phase_id, event_ns),
        "candidate_sha256": common.sha256_bytes(candidate_raw),
        "clock_id": contract["phase_protocol"]["clock_id"],
        "contract_sha256": contract["raw_predicate_contract"]["sha256"],
        "model_slot": "A",
        "prior_phase_result_sha256s": [],
        "quality_corpus_sha256": common.sha256_bytes(corpus_raw),
        "route_lock_sha256": common.sha256_bytes(route_raw),
    }
    return _jsonl([row])


def materialize(
    *,
    output: Path,
    phase_id: str,
    contract_path: Path,
    candidate_path: Path,
    artifact_root_path: Path,
    preparation_path: Path,
    quality_corpus_path: Path,
    runtime_plan_path: Path,
    pre_dir: Path,
    confirmation: str,
    timeout_seconds: int,
    runner=None,
    clock_ns: Callable[[], int] = common.monotonic_ns,
) -> dict:
    phase_id = common.validate_phase_id(phase_id)
    common.exact(confirmation, CONFIRMATION, "confirmation")
    common.require(
        type(timeout_seconds) is int and 30 <= timeout_seconds <= 600,
        "E_TIMEOUT",
    )
    common.require(not output.exists(), "E_OUTPUT_EXISTS")
    contract, contract_raw = common.read_canonical(contract_path, "contract")
    candidate, candidate_raw = common.read_canonical(candidate_path, "candidate")
    root, root_raw = common.read_canonical(artifact_root_path, "artifact_root")
    preparation, preparation_raw = common.read_canonical(preparation_path, "preparation")
    plan, plan_raw = common.read_canonical(runtime_plan_path, "runtime_plan")
    corpus_raw = common.read_regular(quality_corpus_path, "quality_corpus")
    common.exact(contract.get("schema"), "s39-cp0-r1-evidence-contract-v2.4", "contract.schema")
    common.exact(candidate.get("schema"), "s39-cp0-r1-candidate-v1", "candidate.schema")
    common.exact(root.get("schema"), "s39-cp0-r1-artifact-root-v2.4", "root.schema")
    common.exact(
        preparation.get("schema"),
        "s39-cp0-r1-reboot-preparation-v2.4",
        "preparation.schema",
    )
    common.exact(plan.get("schema"), "s39-cp0-r1-runtime-bundle-plan-v2.4", "plan.schema")
    common.exact(common.sha256_bytes(corpus_raw), contract["quality_corpus"]["sha256"], "corpus.sha256")
    common.exact(len(corpus_raw), contract["quality_corpus"]["bytes"], "corpus.bytes")
    common.exact(root["runtime_bundle_plan_sha256"], common.sha256_bytes(plan_raw), "root.plan")
    common.exact(
        preparation["runtime_bundle_plan_sha256"],
        common.sha256_bytes(plan_raw),
        "preparation.plan",
    )
    common.exact(
        preparation["artifact_root_sha256"],
        common.sha256_bytes(root_raw),
        "preparation.root",
    )
    boot_ids = {
        endpoint: preparation["devices"][endpoint][
            "host_boot_id" if endpoint == "cuda" else "boot_id"
        ]
        for endpoint in ("cuda", "op12", "op15")
    }
    for endpoint, value in boot_ids.items():
        common.require(common.UUID_RE.fullmatch(value) is not None, f"E_BOOT_ID: {endpoint}")
    event_ns = clock_ns()
    common.require(preparation["completed_ns"] <= event_ns, "E_LOCK_BEFORE_PREPARATION")
    common.require(
        event_ns - root["completed_ns"]
        <= contract["gates"]["artifact_root_maximum_age_ns"],
        "E_ARTIFACT_ROOT_STALE",
    )
    result = {
        "artifact_root_sha256": common.sha256_bytes(root_raw),
        "candidate_sha256": common.sha256_bytes(candidate_raw),
        "contract_sha256": common.sha256_bytes(contract_raw),
        "device_boot_ids": boot_ids,
        "event_ns": event_ns,
        "model_id": common.MODEL_ID,
        "phase": common.PHASE,
        "phase_id": phase_id,
        "preparation_sha256": common.sha256_bytes(preparation_raw),
        "quality_corpus_sha256": common.sha256_bytes(corpus_raw),
        "runtime_bundle_plan_sha256": common.sha256_bytes(plan_raw),
        "schema": "s39-cp0-r1-phase-lock-v2.4",
    }
    capture = common.capture_preflight(
        runner or common.SubprocessRunner(),
        contract,
        root,
        timeout_seconds,
        clock_ns,
    )
    raw_root = pre_dir / "raw"
    common.require(not raw_root.exists(), "E_PRE_RAW_EXISTS")
    raw_root.mkdir(parents=True, exist_ok=False)
    preflight_raw = _preflight_raw(capture, phase_id)
    corpus_role_raw = _corpus_raw(
        corpus_raw,
        phase_id,
        event_ns - 200,
    )
    route_raw = _route_lock_raw(
        contract,
        candidate,
        phase_id,
        event_ns - 100,
    )
    lock_raw = _phase_lock_raw(
        contract,
        candidate_raw,
        phase_id,
        event_ns,
        corpus_role_raw,
        route_raw,
    )
    for name, raw in (
        ("phase-lock.jsonl", lock_raw),
        ("phase-preflight.jsonl", preflight_raw),
        ("quality-corpus.jsonl", corpus_role_raw),
        ("route-lock.jsonl", route_raw),
    ):
        common.write_raw_new(raw_root / name, raw)
    common.write_new(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--quality-corpus", type=Path, required=True)
    parser.add_argument("--runtime-plan", type=Path, required=True)
    parser.add_argument("--pre-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        materialize(
            output=args.output,
            phase_id=args.phase_id,
            contract_path=args.contract,
            candidate_path=args.candidate,
            artifact_root_path=args.root,
            preparation_path=args.preparation,
            quality_corpus_path=args.quality_corpus,
            runtime_plan_path=args.runtime_plan,
            pre_dir=args.pre_dir,
            confirmation=args.confirm if args.execute else None,
            timeout_seconds=args.timeout_seconds,
        )
        return 0
    except (OSError, ValueError, common.ProductionError) as error:
        print(f"V24_PHASE_LOCK_REFUSED: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
