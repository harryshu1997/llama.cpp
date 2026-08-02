#!/usr/bin/env python3
"""Fan V2.4 producer captures into the raw authority inputs."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
V24 = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(V24) not in sys.path:
    sys.path.insert(0, str(V24))

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
    HERE / "production_common_v1.py",
)
authority = _load_source(
    "s39_v24_authority",
    V24 / "cp0_r1_evidence_v24.py",
)


PRE_RAW = {
    "phase.lock": "phase-lock.jsonl",
    "phase.preflight": "phase-preflight.jsonl",
    "quality.corpus": "quality-corpus.jsonl",
    f"model.{common.MODEL_ID}.route_lock": "route-lock.jsonl",
}
JOINT_FIELDS = copy.deepcopy(authority.JOINT_ROLE_FIELDS)
MONOLITHIC_ROLE = f"model.{common.MODEL_ID}.oracle.cuda_monolithic"


def _raw_rows(rows: Any, role: str, phase_id: str) -> bytes:
    common.require(type(rows) is list and bool(rows), f"E_ROWS: {role}")
    result = bytearray()
    previous = None
    for index, row in enumerate(rows):
        common.require(type(row) is dict, f"E_ROW_TYPE: {role}[{index}]")
        event = common.integer(row.get("event_ns"), f"{role}[{index}].event_ns", 1)
        if previous is not None:
            common.require(previous <= event, f"E_EVENT_ORDER: {role}")
        previous = event
        result.extend(
            common.canonical_bytes(
                {
                    "acquisition_id": phase_id,
                    **row,
                    "phase": common.PHASE,
                    "phase_id": phase_id,
                    "role": role,
                }
            )
        )
    return bytes(result)


def _artifact(path: Path, role: str, raw: bytes, *, manifest: bool) -> dict[str, Any]:
    return {
        "bytes": len(raw),
        **({"format": "CANONICAL_ASCII_JSONL"} if manifest else {}),
        "path": str(path),
        "role": role,
        "sha256": common.sha256_bytes(raw),
    }


def _require_pre_raw(pre_dir: Path) -> dict[str, bytes]:
    missing = []
    result = {}
    for role, name in PRE_RAW.items():
        path = (pre_dir / "raw" / name).resolve()
        if not path.is_file():
            missing.append(f"{role}={path}")
            continue
        result[role] = common.read_regular(path, f"pre_raw.{role}")
    common.require(
        not missing,
        "E_PRODUCTION_INPUTS_MISSING: " + ",".join(missing),
    )
    return result


def _first_event(raw: bytes, field: str) -> int:
    line = raw.splitlines(keepends=True)[0]
    value = common.parse_json(line, field)
    return common.integer(value.get("event_ns"), f"{field}.event_ns", 1)


def _process_record(
    captured: dict[str, Any],
    bundle: dict[str, Any],
    root_components: dict[str, dict[str, Any]],
    evidence_role: str,
    evidence_sha256: str,
    model_mapping: dict[str, Any] | None,
) -> dict[str, Any]:
    field = f"runtime_process.{bundle['bundle_id']}"
    required = {
        "boot_id",
        "bundle_id",
        "endpoint",
        "launcher_path",
        "loaded_repo_component_ids",
        "observed_ns",
        "pid",
        "process_swap_bytes",
        "start_ticks",
    }
    missing = sorted(required - set(captured))
    common.require(not missing, f"E_PRODUCER_FIELDS_MISSING: {field}: {missing}")
    identity = {
        "bundle_id": bundle["bundle_id"],
        "components": [
            {
                "component_id": component_id,
                "path": root_components[component_id]["path"],
                "sha256": root_components[component_id]["sha256"],
                "stat": root_components[component_id]["stat"],
            }
            for component_id in bundle["required_component_ids"]
        ],
        "endpoint": bundle["endpoint"],
        "launcher_component_id": bundle["launcher_component_id"],
        "process_role": bundle["process_role"],
        "schema": "s39-cp0-r1-runtime-bundle-root-identity-v2.4",
    }
    return {
        "boot_id": captured["boot_id"],
        "bundle_id": bundle["bundle_id"],
        "bundle_sha256": common.sha256_bytes(common.canonical_bytes(identity)),
        "endpoint": captured["endpoint"],
        "evidence_role": evidence_role,
        "evidence_sha256": evidence_sha256,
        "launcher_component_id": bundle["launcher_component_id"],
        "launcher_path": captured["launcher_path"],
        "loaded_repo_component_ids": captured["loaded_repo_component_ids"],
        "model_mapping": model_mapping,
        "observed_ns": captured["observed_ns"],
        "pid": captured["pid"],
        "process_swap_bytes": captured["process_swap_bytes"],
        "start_ticks": captured["start_ticks"],
    }


def fan_in(
    *,
    runtime_output: Path,
    acquisition_output: Path,
    bundle_root: Path,
    pre_dir: Path,
    acquisition_started_ns: int,
    contract_path: Path,
    candidate_path: Path,
    runtime_plan_path: Path,
    artifact_root_path: Path,
    preparation_path: Path,
    phase_lock_path: Path,
    fresh_path: Path,
    cuda_monolithic_path: Path,
    joint_phone_cuda_path: Path,
    clock_ns: Callable[[], int] = common.monotonic_ns,
) -> tuple[dict[str, Any], dict[str, Any]]:
    common.require(
        not runtime_output.exists()
        and not acquisition_output.exists(),
        "E_OUTPUT_EXISTS",
    )
    if bundle_root.exists():
        common.require(
            bundle_root.is_dir()
            and not bundle_root.is_symlink()
            and not any(bundle_root.iterdir()),
            "E_BUNDLE_ROOT_NOT_EMPTY",
        )
    common.integer(acquisition_started_ns, "acquisition_started_ns", 1)
    contract, contract_raw = common.read_canonical(contract_path, "contract")
    candidate, candidate_raw = common.read_canonical(candidate_path, "candidate")
    plan, plan_raw = common.read_canonical(runtime_plan_path, "runtime_plan")
    root, root_raw = common.read_canonical(artifact_root_path, "artifact_root")
    preparation, preparation_raw = common.read_canonical(preparation_path, "preparation")
    lock, lock_raw = common.read_canonical(phase_lock_path, "phase_lock")
    fresh, fresh_raw = common.read_canonical(fresh_path, "fresh")
    monolithic, monolithic_raw = common.read_canonical(cuda_monolithic_path, "cuda_monolithic")
    joint, joint_raw = common.read_canonical(joint_phone_cuda_path, "joint_phone_cuda")
    common.exact(contract.get("schema"), "s39-cp0-r1-evidence-contract-v2.4", "contract.schema")
    common.exact(plan.get("schema"), "s39-cp0-r1-runtime-bundle-plan-v2.4", "plan.schema")
    common.exact(monolithic.get("schema"), "s39-cp0-r1-v24-cuda-monolithic-raw-v1", "monolithic.schema")
    common.exact(joint.get("schema"), "s39-cp0-r1-v24-joint-phone-cuda-raw-v1", "joint.schema")
    phase_id = common.validate_phase_id(lock["phase_id"])
    common.exact(monolithic["phase_id"], phase_id, "monolithic.phase_id")
    common.exact(joint["phase_id"], phase_id, "joint.phase_id")
    common.require(
        acquisition_started_ns <= min(monolithic["started_ns"], joint["started_ns"]),
        "E_PRODUCER_BEFORE_ACQUISITION",
    )
    pre_raw = _require_pre_raw(pre_dir)
    bundle_root.mkdir(parents=True, exist_ok=True)
    raw_dir = bundle_root / "raw"
    raw_dir.mkdir()
    manifest_artifacts = []
    acquisition_artifacts = []
    raw_by_role = dict(pre_raw)
    raw_by_role[MONOLITHIC_ROLE] = _raw_rows(
        monolithic["oracle_cuda_monolithic_rows"],
        MONOLITHIC_ROLE,
        phase_id,
    )
    for role, field in JOINT_FIELDS.items():
        raw_by_role[role] = _raw_rows(joint[field], role, phase_id)
    for index, role in enumerate(sorted(raw_by_role)):
        raw = raw_by_role[role]
        relative = Path("raw") / f"{index:02d}.jsonl"
        common.write_raw_new(bundle_root / relative, raw)
        manifest_artifacts.append(_artifact(relative, role, raw, manifest=True))
        if role not in authority.PRE_ACQUISITION_ROLES:
            acquisition_artifacts.append(_artifact(relative, role, raw, manifest=False))
    for role, raw, name in (
        ("capture.cuda_monolithic", monolithic_raw, "cuda-monolithic.json"),
        ("capture.joint_phone_cuda", joint_raw, "joint-phone-cuda.json"),
    ):
        relative = Path("raw") / name
        common.write_raw_new(bundle_root / relative, raw)
        acquisition_artifacts.append(_artifact(relative, role, raw, manifest=False))
    receipt_digests = {
        "capture.cuda_monolithic": common.sha256_bytes(monolithic_raw),
        "capture.joint_phone_cuda": common.sha256_bytes(joint_raw),
    }
    bundles = {value["bundle_id"]: value for value in plan["bundles"]}
    components = {value["component_id"]: value for value in root["components"]}
    captured = {
        value["bundle_id"]: value
        for value in joint["runtime_processes"]
    }
    captured["cuda_monolithic"] = monolithic["runtime_process"]
    model_binding = monolithic["runtime_model_binding"]
    model_mapping = {
        "argv": model_binding["argv"],
        "environment": plan["cuda_monolithic_launch"]["env"],
        "model_mapping_rows": model_binding["model_mapping_rows"],
        "model_file_type": contract["cuda_monolithic_identity"]["expected_file_type"],
        "model_path": model_binding["model_path"],
        "model_sha256": model_binding["model_sha256"],
        "other_gguf_mapping_paths": model_binding["other_gguf_mapping_paths"],
        "post_stat": model_binding["model_stat"],
        "pre_stat": model_binding["model_stat"],
    }
    processes = []
    for bundle_id in sorted(bundles):
        role = authority.PROCESS_EVIDENCE_ROLES[bundle_id]
        processes.append(
            _process_record(
                captured[bundle_id],
                bundles[bundle_id],
                components,
                role,
                receipt_digests[role],
                model_mapping if bundle_id == "cuda_monolithic" else None,
            )
        )
    started_ns = min(monolithic["started_ns"], joint["started_ns"])
    completed_ns = max(monolithic["completed_ns"], joint["completed_ns"])
    phone_after = {}
    for phone in ("op12", "op15"):
        probe = joint["phone_evidence"]["raw_probes"][phone]
        after = probe["after"]
        phone_after[phone] = {
            "boot_id": after["boot_id"],
            "observed_ns": probe["after_ns"],
            "system_swap_used_bytes": after["system_swap_used_bytes"],
        }
    runtime = {
        "artifact_root_sha256": common.sha256_bytes(root_raw),
        "completed_ns": completed_ns,
        "fresh_readiness_sha256": common.sha256_bytes(fresh_raw),
        "phase": common.PHASE,
        "phase_id": phase_id,
        "phone_after": phone_after,
        "processes": processes,
        "runtime_bundle_plan_sha256": common.sha256_bytes(plan_raw),
        "schema": "s39-cp0-r1-runtime-identity-v2.4",
        "started_ns": started_ns,
    }
    runtime_raw = common.canonical_bytes(runtime)
    phase_closed_ns = clock_ns()
    common.require(completed_ns <= phase_closed_ns, "E_FAN_IN_INTERVAL")
    manifest = {
        "acquisition_started_ns": acquisition_started_ns,
        "artifacts": manifest_artifacts,
        "candidate_sha256": common.sha256_bytes(candidate_raw),
        "clock_id": "HOST_MONOTONIC_RAW",
        "contract_sha256": contract["raw_predicate_contract"]["sha256"],
        "phase": common.PHASE,
        "phase_closed_ns": phase_closed_ns,
        "phase_id": phase_id,
        "phase_opened_ns": min(
            _first_event(raw, f"pre_raw.{role}")
            for role, raw in pre_raw.items()
        ),
        "schema": "s39-cp0-r1-evidence-bundle-v2.1",
    }
    manifest_raw = common.canonical_bytes(manifest)
    acquisition = {
        "artifact_root_sha256": common.sha256_bytes(root_raw),
        "artifacts": sorted(acquisition_artifacts, key=lambda value: value["role"]),
        "candidate_sha256": common.sha256_bytes(candidate_raw),
        "completed_ns": phase_closed_ns,
        "contract_sha256": common.sha256_bytes(contract_raw),
        "fresh_readiness_sha256": common.sha256_bytes(fresh_raw),
        "phase": common.PHASE,
        "phase_id": phase_id,
        "phase_lock_sha256": common.sha256_bytes(lock_raw),
        "preparation_sha256": common.sha256_bytes(preparation_raw),
        "raw_manifest_name": authority.RAW_MANIFEST_NAME,
        "raw_manifest_sha256": common.sha256_bytes(manifest_raw),
        "raw_predicate_contract_sha256": contract["raw_predicate_contract"]["sha256"],
        "runtime_bundle_plan_sha256": common.sha256_bytes(plan_raw),
        "runtime_identity_sha256": common.sha256_bytes(runtime_raw),
        "schema": "s39-cp0-r1-a-only-acquisition-v2.4",
        "started_ns": acquisition_started_ns,
        "status": "RAW_CAPTURE_COMPLETE_UNEVALUATED",
    }
    common.write_raw_new(bundle_root / authority.RAW_MANIFEST_NAME, manifest_raw)
    common.write_new(runtime_output, runtime)
    common.write_new(acquisition_output, acquisition)
    return runtime, acquisition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--pre-dir", type=Path, required=True)
    parser.add_argument("--started", type=int, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--runtime-plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--phase-lock", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--mono", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        fan_in(
            runtime_output=args.runtime,
            acquisition_output=args.acquisition,
            bundle_root=args.bundle_root,
            pre_dir=args.pre_dir,
            acquisition_started_ns=args.started,
            contract_path=args.contract,
            candidate_path=args.candidate,
            runtime_plan_path=args.runtime_plan,
            artifact_root_path=args.root,
            preparation_path=args.preparation,
            phase_lock_path=args.phase_lock,
            fresh_path=args.fresh,
            cuda_monolithic_path=args.mono,
            joint_phone_cuda_path=args.joint,
        )
        return 0
    except (OSError, ValueError, common.ProductionError) as error:
        print(f"V24_FAN_IN_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
