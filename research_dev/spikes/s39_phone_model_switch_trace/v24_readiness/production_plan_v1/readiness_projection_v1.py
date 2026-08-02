#!/usr/bin/env python3
"""Validate the complete V2.4 chain immediately before acquisition."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys
import types


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


def project(
    *,
    acquisition_started_ns: int,
    contract_path: Path,
    candidate_path: Path,
    runtime_plan_path: Path,
    tokenizer_plan_path: Path,
    token_history_path: Path,
    artifact_root_path: Path,
    preparation_path: Path,
    phase_lock_path: Path,
    fresh_path: Path,
) -> dict:
    common.integer(acquisition_started_ns, "acquisition_started_ns", 1)
    contract, contract_raw, candidate, candidate_raw = authority.validate_inputs(
        contract_path,
        candidate_path,
    )
    plan, plan_raw = authority.common.read_canonical(runtime_plan_path)
    plan_derived = authority.validate_runtime_plan(
        plan,
        contract,
        contract_raw,
        candidate_raw,
    )
    tokenizer, tokenizer_raw = authority.common.read_canonical(tokenizer_plan_path)
    authority.validate_tokenizer_plan(
        tokenizer,
        tokenizer_raw,
        contract,
        candidate,
        plan_derived,
    )
    history, history_raw = authority.common.read_canonical(token_history_path)
    authority.validate_token_history(
        history,
        history_raw,
        contract,
        candidate,
        plan_derived,
    )
    root, root_raw = authority.common.read_canonical(artifact_root_path)
    root_derived = authority.validate_artifact_root(
        root,
        contract,
        contract_raw,
        candidate,
        candidate_raw,
        plan_raw,
        plan_derived,
        history_raw,
        tokenizer_raw,
    )
    preparation, preparation_raw = authority.common.read_canonical(preparation_path)
    preparation_derived = authority.validate_preparation(
        preparation,
        preparation_raw,
        contract,
        root_raw,
        root_derived["completed_ns"],
        plan_raw,
    )
    lock, lock_raw = authority.common.read_canonical(phase_lock_path)
    lock_derived = authority.validate_phase_lock(
        lock,
        lock_raw,
        contract,
        contract_raw,
        candidate_raw,
        root_raw,
        root_derived["completed_ns"],
        preparation_raw,
        preparation_derived,
        plan_raw,
    )
    fresh, fresh_raw = authority.common.read_canonical(fresh_path)
    authority.validate_fresh(
        fresh,
        fresh_raw,
        contract,
        lock_raw,
        lock_derived,
        root_raw,
        root_derived,
        preparation_raw,
        plan_raw,
        plan_derived,
        acquisition_started_ns,
    )
    return {
        "acquisition_started_ns": acquisition_started_ns,
        "artifact_root_sha256": common.sha256_bytes(root_raw),
        "fresh_readiness_sha256": common.sha256_bytes(fresh_raw),
        "phase_id": lock_derived["phase_id"],
        "runtime_bundle_plan_sha256": common.sha256_bytes(plan_raw),
        "schema": "s39-cp0-r1-v24-pre-acquisition-projection-v1",
        "status": "V2_4_PRE_ACQUISITION_READINESS_PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--started", type=int, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--runtime-plan", type=Path, required=True)
    parser.add_argument("--tokenizer-plan", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--phase-lock", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = project(
            acquisition_started_ns=args.started,
            contract_path=args.contract,
            candidate_path=args.candidate,
            runtime_plan_path=args.runtime_plan,
            tokenizer_plan_path=args.tokenizer_plan,
            token_history_path=args.history,
            artifact_root_path=args.root,
            preparation_path=args.preparation,
            phase_lock_path=args.phase_lock,
            fresh_path=args.fresh,
        )
        print(common.canonical_bytes(value).decode("ascii"), end="")
        return 0
    except (OSError, ValueError, common.ProductionError) as error:
        print(f"V24_READINESS_PROJECTION_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
