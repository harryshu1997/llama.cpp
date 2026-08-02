#!/usr/bin/env python3
"""Materialize the current V2.4 A_ONLY production config and pinned plan."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any


HERE = Path(__file__).resolve().parent
V24 = HERE.parent
S39 = V24.parent
REPO = S39.parents[2]
RESULTS = V24 / "results" / "prephase_20260726T0915Z"


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
orchestration = _load_source(
    "s39_v24_orchestration",
    V24 / "orchestration_v1" / "orchestration_v1.py",
)


INPUT_SCHEMAS = {
    "candidate": "s39-cp0-r1-candidate-v1",
    "contract": "s39-cp0-r1-evidence-contract-v2.4",
    "cuda_monolithic_launch": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
    "cuda_route_launch": "s39-cp0-r1-v24-cuda-route-launch-v1",
    "joint_capture_plan": "s39-cp0-r1-v24-joint-capture-plan-v1",
    "phone_route_launch": "s39-cp0-r1-v24-phone-route-launch-v1",
    "prospective_root": "s39-cp0-r1-v24-prospective-runtime-root-v1",
    "runtime_plan": "s39-cp0-r1-runtime-bundle-plan-v2.4",
    "token_history": "s39-cp0-r1-token-history-v2.4",
    "tokenizer_plan": "s39-cp0-r1-a-only-tokenizer-plan-v2",
}
CAPTURE_STAGE_KINDS = {
    "artifact_root": "artifact_root",
    "fresh_readiness": "fast_fresh_readiness",
    "cuda_monolithic": "cuda_monolithic",
    "joint_phone_cuda": "joint_phone_cuda",
}
TIMEOUTS = {
    "artifact_root": 3600,
    "preparation": 1800,
    "phase_lock": 60,
    "identity_binding": 60,
    "fresh_readiness": 300,
    "readiness_projection": 300,
    "cuda_monolithic": 7200,
    "joint_phone_cuda": 7200,
    "fan_in": 300,
    "authority": 1800,
}


def _no_v23(value: Any, field: str) -> None:
    if type(value) is str:
        common.require("v23_readiness" not in value, f"E_STALE_V23: {field}")
    elif type(value) is list:
        for index, item in enumerate(value):
            _no_v23(item, f"{field}[{index}]")
    elif type(value) is dict:
        for key, item in value.items():
            _no_v23(item, f"{field}.{key}")


def _missing(paths: dict[str, Path]) -> None:
    values = [
        f"{name}={path}"
        for name, path in sorted(paths.items())
        if not path.is_file()
    ]
    common.require(
        not values,
        "E_PRODUCTION_INPUTS_MISSING: " + ",".join(values),
    )


def _read_inputs(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    values = {}
    for name, expected in INPUT_SCHEMAS.items():
        value, _ = common.read_canonical(paths[name], f"input.{name}")
        common.exact(value.get("schema"), expected, f"input.{name}.schema")
        if name in {
            "cuda_monolithic_launch",
            "cuda_route_launch",
            "joint_capture_plan",
            "phone_route_launch",
            "runtime_plan",
        }:
            _no_v23(value, f"input.{name}")
        values[name] = value
    corpus_raw = common.read_regular(paths["quality_corpus"], "input.quality_corpus")
    contract = values["contract"]
    common.exact(
        common.sha256_bytes(corpus_raw),
        contract["quality_corpus"]["sha256"],
        "quality_corpus.sha256",
    )
    common.exact(
        len(corpus_raw),
        contract["quality_corpus"]["bytes"],
        "quality_corpus.bytes",
    )
    return values


def _capture_paths(runtime: dict[str, Any]) -> dict[str, Path]:
    components = {
        value["component_id"]: value
        for value in runtime["components"]
    }
    captures = {
        value["kind"]: value
        for value in runtime["capture_entrypoints"]
    }
    common.exact(set(captures), set(CAPTURE_STAGE_KINDS.values()), "runtime.capture_kinds")
    return {
        stage: Path(components[captures[kind]["component_id"]]["path"])
        for stage, kind in CAPTURE_STAGE_KINDS.items()
    }


def _support_paths(
    contract: dict[str, Any],
) -> tuple[list[str], dict[str, list[str]]]:
    authority = contract["exit_authority"]["support"]
    authority_paths = [
        str((S39 / record["path"]).resolve())
        for _, record in sorted(authority.items())
    ]
    requirements = contract["orchestration_requirements"]
    support = requirements["support"]
    stages = {
        stage: sorted(
            str((S39 / support[name]["path"]).resolve())
            for name in names
        )
        for stage, names in requirements["stage_support"].items()
    }
    return authority_paths, stages


def _base(
    entrypoint: Path,
    argv: list[str],
    support_files: list[str],
    timeout: int,
) -> dict[str, Any]:
    environment = {
        "ADB_SERVER_PORT": str(common.PHONE_ADB_PORT),
        "ANDROID_ADB_SERVER_PORT": str(common.PHONE_ADB_PORT),
        "LC_ALL": "C",
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "S39_CUDA_SSH_TARGET": common.CUDA_SSH_TARGET,
    }
    return {
        "argv_template": argv,
        "cwd": str(REPO),
        "entrypoint": str(entrypoint),
        "environment": environment,
        "support_files": support_files,
        "timeout_seconds": timeout,
    }


def _commands(
    contract: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    capture = _capture_paths(runtime)
    authority_entrypoint = (S39 / contract["exit_authority"]["entrypoint"]["path"]).resolve()
    authority_support, stage_support = _support_paths(contract)
    values = {
        "artifact_root": _base(
            capture["artifact_root"],
            [
                "--output", "{artifact_root}",
                "--contract", "{contract}",
                "--candidate", "{candidate}",
                "--runtime-plan", "{runtime_plan}",
                "--history", "{token_history}",
                "--tokenizer-plan", "{tokenizer_plan}",
                "--cuda-ssh-target", common.CUDA_SSH_TARGET,
                "--phone-adb-port", str(common.PHONE_ADB_PORT),
                "--confirm", "RUN_V24_ARTIFACT_ROOT_A_ONLY",
            ],
            [],
            TIMEOUTS["artifact_root"],
        ),
        "preparation": _base(
            HERE / "preparation_v1.py",
            [
                "--output", "{preparation}",
                "--contract", "{contract}",
                "--root", "{artifact_root}",
                "--runtime-plan", "{runtime_plan}",
                "--confirm", "RUN_V24_REBOOT_PREPARATION_A_ONLY",
            ],
            stage_support["preparation"],
            TIMEOUTS["preparation"],
        ),
        "phase_lock": _base(
            HERE / "phase_lock_v1.py",
            [
                "--output", "{phase_lock}",
                "--phase-id", "{phase_id}",
                "--contract", "{contract}",
                "--candidate", "{candidate}",
                "--root", "{artifact_root}",
                "--preparation", "{preparation}",
                "--quality-corpus", "{quality_corpus}",
                "--runtime-plan", "{runtime_plan}",
                "--pre-dir", "{pre_dir}",
                "--execute",
                "--confirm", "RUN_V24_PHASE_LOCK_PREFLIGHT_A_ONLY",
            ],
            stage_support["phase_lock"],
            TIMEOUTS["phase_lock"],
        ),
        "identity_binding": _base(
            HERE / "identity_binding_v1.py",
            [
                "--prospective-root", "{prospective_root}",
                "--contract", "{contract}",
                "--preparation", "{preparation}",
                "--phase-lock", "{phase_lock}",
                "--prospective-cuda-route-launch", "{cuda_route_launch}",
                "--bound-cuda-route-launch", "{bound_cuda_route_launch}",
                "--prospective-joint-capture-plan", "{joint_capture_plan}",
                "--bound-joint-capture-plan", "{bound_joint_capture_plan}",
                "--prospective-phone-route-launch", "{phone_route_launch}",
                "--bound-phone-route-launch", "{bound_phone_route_launch}",
                "--prospective-runtime-plan", "{runtime_plan}",
                "--bound-runtime-plan", "{bound_runtime_plan}",
                "--receipt", "{identity_binding_receipt}",
                "--bound-root", "{bound_root}",
            ],
            stage_support["identity_binding"],
            TIMEOUTS["identity_binding"],
        ),
        "fresh_readiness": _base(
            capture["fresh_readiness"],
            [
                "--output", "{fresh}",
                "--phase-id", "{phase_id}",
                "--contract", "{contract}",
                "--root", "{artifact_root}",
                "--preparation", "{preparation}",
                "--phase-lock", "{phase_lock}",
                "--runtime-plan", "{bound_runtime_plan}",
                "--cuda-ssh-target", common.CUDA_SSH_TARGET,
                "--phone-adb-port", str(common.PHONE_ADB_PORT),
                "--confirm", "RUN_V24_FAST_FRESH_READINESS_A_ONLY",
            ],
            [],
            TIMEOUTS["fresh_readiness"],
        ),
        "readiness_projection": _base(
            HERE / "readiness_projection_v1.py",
            [
                "--started", "{acquisition_started_ns}",
                "--contract", "{contract}",
                "--candidate", "{candidate}",
                "--runtime-plan", "{bound_runtime_plan}",
                "--tokenizer-plan", "{tokenizer_plan}",
                "--history", "{token_history}",
                "--root", "{artifact_root}",
                "--preparation", "{preparation}",
                "--phase-lock", "{phase_lock}",
                "--fresh", "{fresh}",
            ],
            stage_support["readiness_projection"],
            TIMEOUTS["readiness_projection"],
        ),
        "cuda_monolithic": _base(
            capture["cuda_monolithic"],
            [
                "--output", "{cuda_monolithic}",
                "--phase-id", "{phase_id}",
                "--pre-dir", "{pre_dir}",
                "--started", "{acquisition_started_ns}",
                "--plan", "{orchestration_plan_sha256}",
                "--mechanism-commands-sha256", "{mechanism_commands_sha256}",
                "--model-sha256", "{model_sha256}",
                "--histories", "{token_history}",
                "--histories-sha256", "{token_history_sha256}",
                "--launch-plan", "{cuda_monolithic_launch}",
                "--launch-plan-sha256", "{cuda_monolithic_launch_sha256}",
                "--execute",
                "--confirm", "RUN_V24_CUDA_MONOLITHIC_A_ONLY",
            ],
            [],
            TIMEOUTS["cuda_monolithic"],
        ),
        "joint_phone_cuda": _base(
            capture["joint_phone_cuda"],
            [
                "--capture-plan", "{bound_joint_capture_plan}",
                "--capture-plan-sha256", "{bound_joint_capture_plan_sha256}",
                "--output", "{joint_phone_cuda}",
                "--phase-id", "{phase_id}",
                "--pre-dir", "{pre_dir}",
                "--acquisition-started-ns", "{acquisition_started_ns}",
                "--command-plan-sha256", "{orchestration_plan_sha256}",
                "--execute",
                "--confirm", "RUN_V24_JOINT_PHONE_CUDA_A_ONLY",
            ],
            [],
            TIMEOUTS["joint_phone_cuda"],
        ),
        "fan_in": _base(
            HERE / "fan_in_v1.py",
            [
                "--runtime", "{runtime_identity}",
                "--acquisition", "{acquisition}",
                "--bundle-root", "{bundle_root}",
                "--pre-dir", "{pre_dir}",
                "--started", "{acquisition_started_ns}",
                "--contract", "{contract}",
                "--candidate", "{candidate}",
                "--runtime-plan", "{bound_runtime_plan}",
                "--root", "{artifact_root}",
                "--preparation", "{preparation}",
                "--phase-lock", "{phase_lock}",
                "--fresh", "{fresh}",
                "--mono", "{cuda_monolithic}",
                "--joint", "{joint_phone_cuda}",
            ],
            stage_support["fan_in"],
            TIMEOUTS["fan_in"],
        ),
        "authority": _base(
            authority_entrypoint,
            [
                "--contract", "{contract}",
                "--candidate", "{candidate}",
                "--runtime-plan", "{bound_runtime_plan}",
                "--tokenizer-plan", "{tokenizer_plan}",
                "--token-history", "{token_history}",
                "--artifact-root", "{artifact_root}",
                "--preparation", "{preparation}",
                "--phase-lock", "{phase_lock}",
                "--fresh", "{fresh}",
                "--runtime-identity", "{runtime_identity}",
                "--acquisition", "{acquisition}",
                "--bundle-root", "{bundle_root}",
                "--orchestration-plan", "{orchestration_plan}",
                "--prospective-root", "{prospective_root}",
                "--bound-root", "{bound_root}",
                "--identity-binding-receipt", "{identity_binding_receipt}",
                "--identity-binding-stage-receipt", "{identity_binding_stage_receipt}",
            ],
            authority_support,
            TIMEOUTS["authority"],
        ),
    }
    return values


def _validate_topology(values: dict[str, dict[str, Any]]) -> None:
    contract = values["contract"]
    runtime = values["runtime_plan"]
    phone = values["phone_route_launch"]
    cuda = values["cuda_route_launch"]
    joint = values["joint_capture_plan"]
    common.exact(contract["phase_protocol"]["phase"], common.PHASE, "contract.phase")
    common.exact(runtime["phase"], common.PHASE, "runtime.phase")
    common.exact(joint["phase"], common.PHASE, "joint.phase")
    common.exact(
        values["prospective_root"]["identity_placeholders"],
        common.UNBOUND_BOOT_IDS,
        "prospective.identity_placeholders",
    )
    common.exact(
        values["prospective_root"]["network_placeholders"],
        common.UNBOUND_PHONE_NETWORK,
        "prospective.network_placeholders",
    )
    for phone_name in ("op12", "op15"):
        common.exact(
            phone["phones"][phone_name]["boot_id"],
            common.UNBOUND_BOOT_IDS[phone_name],
            f"prospective.phone.{phone_name}.boot_id",
        )
        for key, expected in common.UNBOUND_PHONE_NETWORK[phone_name].items():
            common.exact(
                phone["phones"][phone_name][key],
                expected,
                f"prospective.phone.{phone_name}.{key}",
            )
    common.exact(
        phone["phones"]["op15"]["direct_peer_ipv4"],
        phone["phones"]["op12"]["local_ipv4"],
        "direct.op15_to_op12",
    )
    common.exact(
        phone["phones"]["op12"]["direct_peer_ipv4"],
        phone["phones"]["op15"]["local_ipv4"],
        "direct.op12_to_op15",
    )
    common.require(
        values["prospective_root"]["desktop_control"]["cuda_ssh_target"]
        == common.CUDA_SSH_TARGET,
        "E_CUDA_SSH_TARGET_UNBOUND",
    )
    adb_tokens = [
        value
        for value in _walk_strings(phone)
        if value == str(common.PHONE_ADB_PORT)
    ]
    common.require(bool(adb_tokens), "E_PHONE_ADB_PORT_UNBOUND")


def _walk_strings(value: Any):
    if type(value) is str:
        yield value
    elif type(value) is list:
        for item in value:
            yield from _walk_strings(item)
    elif type(value) is dict:
        for item in value.values():
            yield from _walk_strings(item)


def materialize(
    *,
    input_paths: dict[str, Path],
    phase_id: str,
    run_root: Path,
    config_output: Path,
    plan_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase_id = common.validate_phase_id(phase_id)
    common.require(
        run_root.is_absolute()
        and not run_root.exists()
        and not config_output.exists()
        and not plan_output.exists(),
        "E_OUTPUT_EXISTS",
    )
    _missing(input_paths)
    values = _read_inputs(input_paths)
    _validate_topology(values)
    commands = _commands(values["contract"], values["runtime_plan"])
    source_paths = {
        f"stage.{stage}": Path(value["entrypoint"])
        for stage, value in commands.items()
    }
    for stage, value in commands.items():
        source_paths.update(
            {
                f"support.{stage}[{index}]": Path(path)
                for index, path in enumerate(value["support_files"])
            }
        )
    _missing(source_paths)
    config = {
        "commands": commands,
        "inputs": {
            name: str(input_paths[name])
            for name in orchestration.INPUT_NAMES
        },
        "model_id": common.MODEL_ID,
        "phase": common.PHASE,
        "phase_id": phase_id,
        "run_root": str(run_root),
        "schema": orchestration.CONFIG_SCHEMA,
    }
    _no_v23(config, "config")
    plan = orchestration._build_plan(config)
    common.write_new(config_output, config)
    common.write_new(plan_output, plan)
    return config, plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=S39 / "CP0_R1_CANDIDATE.json")
    parser.add_argument("--contract", type=Path, default=V24 / "CP0_R1_EVIDENCE_CONTRACT_V2_4.json")
    parser.add_argument("--quality-corpus", type=Path, default=S39 / "CP0_R1_MMLU64_CORPUS_V2_2.jsonl")
    parser.add_argument("--token-history", type=Path, default=RESULTS / "token-history.json")
    parser.add_argument("--tokenizer-plan", type=Path, default=RESULTS / "tokenizer-plan.json")
    parser.add_argument("--cuda-monolithic-launch", type=Path, default=RESULTS / "cuda-monolithic-launch.json")
    parser.add_argument("--cuda-route-launch", type=Path, default=RESULTS / "cuda-route-launch.json")
    parser.add_argument("--phone-route-launch", type=Path, default=RESULTS / "phone-route-launch.json")
    parser.add_argument("--prospective-root", type=Path, default=RESULTS / "prospective-runtime-root.json")
    parser.add_argument("--joint-capture-plan", type=Path, default=RESULTS / "joint-capture-plan.json")
    parser.add_argument("--runtime-plan", type=Path, default=RESULTS / "runtime-bundle-plan.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        name: getattr(args, name)
        for name in (
            "candidate",
            "contract",
            "cuda_monolithic_launch",
            "cuda_route_launch",
            "joint_capture_plan",
            "phone_route_launch",
            "prospective_root",
            "quality_corpus",
            "runtime_plan",
            "token_history",
            "tokenizer_plan",
        )
    }
    try:
        materialize(
            input_paths={name: path.resolve() for name, path in paths.items()},
            phase_id=args.phase_id,
            run_root=args.run_root.resolve(),
            config_output=args.config_output.resolve(),
            plan_output=args.plan_output.resolve(),
        )
        return 0
    except (OSError, ValueError, common.ProductionError) as error:
        print(f"V24_PRODUCTION_PLAN_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
