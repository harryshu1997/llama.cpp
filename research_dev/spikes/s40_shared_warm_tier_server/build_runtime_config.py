#!/usr/bin/env python3
"""Build the canonical S40 llama-server warm-tier runtime configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from evidence_common import (
    EvidenceError,
    canonical_bytes,
    digest_file,
    read_json,
    require,
    require_int,
    require_string,
    validate_digest,
)
from validate_inputs import DEFAULT_CONTRACT, validate_contract


PLAN_KEYS = {
    "c3_profile_lock_path",
    "c3_profile_lock_sha256",
    "evidence_root_path",
    "evidence_root_sha256",
    "event_log_path",
    "executors",
    "hot_model_id",
    "mode",
    "run_id",
    "schema",
}
EVIDENCE_ROOT_KEYS = {
    "c3_placement_artifacts",
    "configuration",
    "executor_configs",
    "experiment_contract_sha256",
    "schema",
}
EVIDENCE_EXECUTOR_KEYS = {
    "executor_id",
    "gateway_config_path",
    "gateway_config_sha256",
}
EVIDENCE_PLACEMENT_KEYS = {
    "path",
    "sha256",
}
PROFILE_LOCK_KEYS = {
    "gpu_uuid",
    "measured_gpu_headroom_bytes",
    "models",
    "schema",
    "system_swap_growth_bytes",
}
PROFILE_MODEL_KEYS = {
    "b1_service_pass",
    "b8_service_pass",
    "n_gpu_layers",
    "placement_evidence_sha256",
}
EXECUTOR_TEMPLATE_KEYS = {
    "credits",
    "execute_concurrency",
    "executor_id",
    "kind",
    "output_limit_bytes",
    "queue_capacity",
    "socket_path",
    "timeout_ms",
    "transport",
}
EXECUTOR_KEYS = EXECUTOR_TEMPLATE_KEYS | {
    "executor_instance_id",
    "expected_peer_pid",
    "expected_peer_start_time_ticks",
}
KINDS_BY_MODE = {
    "C1_GPU_ONLY_OPTIMIZED": ("GPU_PRIMARY",),
    "C2_GPU_PLUS_CPU_WARM_EXECUTOR": ("GPU_PRIMARY", "CPU_WARM"),
    "C3_DUAL_PARTIAL_OFFLOAD": ("GPU_CPU_PARTIAL",),
    "T1_PHONE_WARM_TIER": ("GPU_PRIMARY", "PHONE_WARM"),
    "T2_PHONE_NO_PROMOTION": ("GPU_PRIMARY", "PHONE_WARM"),
}
ROLE_BY_KIND = {
    "GPU_PRIMARY": "GPU",
    "CPU_WARM": "CPU",
    "GPU_CPU_PARTIAL": "GPU",
    "PHONE_WARM": "PHONE",
}


def _validate_socket_path(value: Any, field: str) -> str:
    require(
        isinstance(value, str)
        and 1 <= len(value.encode("ascii", errors="ignore")) <= 103
        and value.isascii()
        and all(0x20 <= ord(character) <= 0x7e for character in value),
        f"{field}: invalid socket path",
    )
    require(Path(value).is_absolute(), f"{field}: expected absolute path")
    return value


def _bounded_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    result = require_int(value, field, minimum)
    require(result <= maximum, f"{field}: expected <= {maximum}")
    return result


def _executor(
        value: Any,
        index: int,
        *,
        require_runtime_identity: bool = True) -> dict[str, Any]:
    field = f"executor[{index}]"
    require(isinstance(value, dict), f"{field}: expected object")
    expected_keys = (
        EXECUTOR_KEYS if require_runtime_identity else EXECUTOR_TEMPLATE_KEYS)
    require(set(value) == expected_keys, f"{field}: key set mismatch")
    kind = require_string(value["kind"], f"{field}.kind")
    require(kind in ROLE_BY_KIND, f"{field}: unknown kind")
    require(
        value["transport"] == "UNIX_SOCKET",
        f"{field}.transport: expected UNIX_SOCKET",
    )
    result = {
        "credits": _bounded_int(
            value["credits"], f"{field}.credits", 1, 128),
        "execute_concurrency": _bounded_int(
            value["execute_concurrency"],
            f"{field}.execute_concurrency",
            1,
            128,
        ),
        "executor_id": require_string(
            value["executor_id"], f"{field}.executor_id"),
        "kind": kind,
        "output_limit_bytes": _bounded_int(
            value["output_limit_bytes"],
            f"{field}.output_limit_bytes",
            1024,
            64 * 1024 * 1024,
        ),
        "queue_capacity": _bounded_int(
            value["queue_capacity"], f"{field}.queue_capacity", 1, 4096),
        "socket_path": _validate_socket_path(
            value["socket_path"], f"{field}.socket_path"),
        "timeout_ms": _bounded_int(
            value["timeout_ms"], f"{field}.timeout_ms", 1, 3_600_000),
        "transport": "UNIX_SOCKET",
    }
    if require_runtime_identity:
        instance_id = require_string(
            value["executor_instance_id"],
            f"{field}.executor_instance_id",
        )
        require(
            instance_id.isascii()
            and len(instance_id) <= 256
            and all(0x21 <= ord(character) <= 0x7e
                    for character in instance_id),
            f"{field}.executor_instance_id: invalid value",
        )
        result.update({
            "executor_instance_id": instance_id,
            "expected_peer_pid": _bounded_int(
                value["expected_peer_pid"],
                f"{field}.expected_peer_pid",
                2,
                2**31 - 1,
            ),
            "expected_peer_start_time_ticks": _bounded_int(
                value["expected_peer_start_time_ticks"],
                f"{field}.expected_peer_start_time_ticks",
                1,
                2**64 - 1,
            ),
        })
    return result


def _validate_c3_profile_lock(
        path_value: Any,
        digest_value: Any,
        contract: dict[str, Any]) -> tuple[Path, str]:
    path_text = require_string(
        path_value, "runtime_plan.c3_profile_lock_path")
    path = Path(path_text)
    require(path.is_absolute(), "runtime_plan: C3 lock path must be absolute")
    digest = validate_digest(
        digest_value, "runtime_plan.c3_profile_lock_sha256")
    require(path.is_file(), "runtime_plan: missing C3 profile lock")
    require(
        digest_file(path) == digest,
        "runtime_plan: C3 profile lock SHA-256 mismatch",
    )
    lock = read_json(path, "c3_profile_lock")
    require(
        path.read_bytes() == canonical_bytes(lock),
        "c3_profile_lock: not canonical JSON",
    )
    require(set(lock) == PROFILE_LOCK_KEYS,
            "c3_profile_lock: key set mismatch")
    require(lock["schema"] == "s40-c3-profile-lock-v1",
            "c3_profile_lock: schema mismatch")
    require(
        False,
        "c3_profile_lock: summary-only v1 is not dispatch eligible; "
        "role-tagged raw dual-residency, B1/B8, VRAM, and swap evidence "
        "is required",
    )
    require(
        lock["gpu_uuid"] == contract["runtime_source"]["expected_gpu_uuid"],
        "c3_profile_lock: GPU identity mismatch",
    )
    headroom = require_int(
        lock["measured_gpu_headroom_bytes"],
        "c3_profile_lock.measured_gpu_headroom_bytes",
    )
    require(
        headroom
        >= contract["matrix"]["C3_DUAL_PARTIAL_OFFLOAD"][
            "minimum_gpu_headroom_bytes"
        ],
        "c3_profile_lock: insufficient GPU headroom",
    )
    require(
        require_int(
            lock["system_swap_growth_bytes"],
            "c3_profile_lock.system_swap_growth_bytes",
        ) == 0,
        "c3_profile_lock: nonzero swap growth",
    )
    models = lock["models"]
    expected_models = contract["workload"]["models"]
    require(
        isinstance(models, dict) and set(models) == set(expected_models),
        "c3_profile_lock: model set mismatch",
    )
    maximum_layers = {
        "qwen3-8b-q8_0": 36,
        "qwen3-14b-q4_k_m": 40,
    }
    for model_id in expected_models:
        record = models[model_id]
        field = f"c3_profile_lock.models.{model_id}"
        require(
            isinstance(record, dict)
            and set(record) == PROFILE_MODEL_KEYS,
            f"{field}: key set mismatch",
        )
        layers = require_int(record["n_gpu_layers"], f"{field}.n_gpu_layers", 1)
        require(
            layers <= maximum_layers[model_id],
            f"{field}: n_gpu_layers exceeds model",
        )
        require(
            record["b1_service_pass"] is True
            and record["b8_service_pass"] is True,
            f"{field}: B1/B8 service not qualified",
        )
        validate_digest(
            record["placement_evidence_sha256"],
            f"{field}.placement_evidence_sha256",
        )
    return path, digest


def _validate_evidence_root(
        path_value: Any,
        digest_value: Any,
        *,
        mode: str,
        executor_ids: list[str],
        contract_path: Path,
        c3_profile_lock: dict[str, Any] | None) -> tuple[Path, str]:
    path = Path(require_string(
        path_value, "runtime_plan.evidence_root_path"))
    require(path.is_absolute(),
            "runtime_plan: evidence root path must be absolute")
    digest = validate_digest(
        digest_value, "runtime_plan.evidence_root_sha256")
    require(path.is_file(), "runtime_plan: missing evidence root")
    require(digest_file(path) == digest,
            "runtime_plan: evidence root SHA-256 mismatch")
    root = read_json(path, "evidence_root")
    require(path.read_bytes() == canonical_bytes(root),
            "evidence_root: not canonical JSON")
    require(set(root) == EVIDENCE_ROOT_KEYS,
            "evidence_root: key set mismatch")
    require(
        root["schema"] == "s40-runtime-evidence-root-v1"
        and root["configuration"] == mode
        and root["experiment_contract_sha256"] == digest_file(contract_path),
        "evidence_root: identity mismatch",
    )
    configs = root["executor_configs"]
    require(
        isinstance(configs, list) and len(configs) == len(executor_ids),
        "evidence_root: executor config count mismatch",
    )
    seen: set[str] = set()
    for index, record in enumerate(configs):
        field = f"evidence_root.executor_configs[{index}]"
        require(
            isinstance(record, dict)
            and set(record) == EVIDENCE_EXECUTOR_KEYS,
            f"{field}: key set mismatch",
        )
        executor_id = require_string(
            record["executor_id"], f"{field}.executor_id")
        require(
            executor_id in executor_ids and executor_id not in seen,
            f"{field}: unknown or duplicate executor",
        )
        seen.add(executor_id)
        config_path = Path(require_string(
            record["gateway_config_path"], f"{field}.gateway_config_path"))
        require(config_path.is_absolute() and config_path.is_file(),
                f"{field}: missing gateway config")
        config_sha = validate_digest(
            record["gateway_config_sha256"],
            f"{field}.gateway_config_sha256",
        )
        require(digest_file(config_path) == config_sha,
                f"{field}: gateway config SHA-256 mismatch")
    require(seen == set(executor_ids),
            "evidence_root: executor ID set mismatch")

    placements = root["c3_placement_artifacts"]
    if mode != "C3_DUAL_PARTIAL_OFFLOAD":
        require(placements == {},
                "evidence_root: unexpected C3 placement artifacts")
    else:
        require(
            c3_profile_lock is not None
            and isinstance(placements, dict)
            and set(placements) == set(c3_profile_lock["models"]),
            "evidence_root: C3 placement model set mismatch",
        )
        for model_id, record in placements.items():
            field = f"evidence_root.c3_placement_artifacts.{model_id}"
            require(
                isinstance(record, dict)
                and set(record) == EVIDENCE_PLACEMENT_KEYS,
                f"{field}: key set mismatch",
            )
            placement_path = Path(require_string(
                record["path"], f"{field}.path"))
            require(placement_path.is_absolute() and placement_path.is_file(),
                    f"{field}: missing placement artifact")
            placement_sha = validate_digest(
                record["sha256"], f"{field}.sha256")
            require(digest_file(placement_path) == placement_sha,
                    f"{field}: placement SHA-256 mismatch")
            require(
                placement_sha
                == c3_profile_lock["models"][model_id][
                    "placement_evidence_sha256"],
                f"{field}: profile-lock digest mismatch",
            )
    return path, digest


def validate_runtime_plan_template(
        plan_path: Path,
        contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    validate_contract(contract_path)
    contract = read_json(contract_path, "contract")
    plan = read_json(plan_path, "runtime_plan_template")
    require(
        plan_path.read_bytes() == canonical_bytes(plan),
        "runtime_plan_template: not canonical JSON",
    )
    require(
        isinstance(plan, dict)
        and set(plan) == PLAN_KEYS
        and plan["schema"] == "s40-runtime-config-plan-v2",
        "runtime_plan_template: identity mismatch",
    )
    run_id = require_string(plan["run_id"], "runtime_plan_template.run_id")
    mode = require_string(plan["mode"], "runtime_plan_template.mode")
    require(mode in KINDS_BY_MODE, "runtime_plan_template: unsupported mode")
    event_log_path = require_string(
        plan["event_log_path"], "runtime_plan_template.event_log_path")
    require(
        Path(event_log_path).is_absolute(),
        "runtime_plan_template: event log path must be absolute",
    )
    hot_model = require_string(
        plan["hot_model_id"], "runtime_plan_template.hot_model_id")
    require(
        hot_model in contract["workload"]["models"],
        "runtime_plan_template: unknown hot model",
    )
    values = plan["executors"]
    require(
        isinstance(values, list),
        "runtime_plan_template.executors: expected array",
    )
    executors = [
        _executor(value, index, require_runtime_identity=False)
        for index, value in enumerate(values)
    ]
    require(
        tuple(executor["kind"] for executor in executors)
        == KINDS_BY_MODE[mode],
        "runtime_plan_template: executor topology mismatch",
    )
    ids = [executor["executor_id"] for executor in executors]
    require(
        len(set(ids)) == len(ids),
        "runtime_plan_template: duplicate executor ID",
    )
    for executor in executors:
        require(
            executor["credits"] <= executor["execute_concurrency"],
            "runtime_plan_template: credits exceed execute concurrency",
        )
        require(
            executor["queue_capacity"] >= executor["credits"],
            "runtime_plan_template: queue capacity is below credits",
        )
    for executor in executors[1:]:
        require(
            executors[0]["credits"] >= executor["credits"],
            "runtime_plan_template: GPU credits are below warm executor credits",
        )

    profile_lock_path = plan["c3_profile_lock_path"]
    profile_lock = plan["c3_profile_lock_sha256"]
    profile_lock_record = None
    if mode == "C3_DUAL_PARTIAL_OFFLOAD":
        profile_path, profile_lock = _validate_c3_profile_lock(
            profile_lock_path, profile_lock, contract)
        profile_lock_path = str(profile_path)
        profile_lock_record = read_json(profile_path, "c3_profile_lock")
    else:
        require(
            profile_lock is None and profile_lock_path is None,
            "runtime_plan_template: unexpected C3 profile lock",
        )
    evidence_root_path, evidence_root_sha256 = _validate_evidence_root(
        plan["evidence_root_path"],
        plan["evidence_root_sha256"],
        mode=mode,
        executor_ids=ids,
        contract_path=contract_path,
        c3_profile_lock=profile_lock_record,
    )
    return {
        "evidence_root_path": str(evidence_root_path),
        "evidence_root_sha256": evidence_root_sha256,
        "executors": executors,
        "mode": mode,
        "plan": plan,
        "profile_lock_path": profile_lock_path,
        "profile_lock_sha256": profile_lock,
        "run_id": run_id,
    }


def build_runtime_config(
        plan_path: Path,
        contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    validate_contract(contract_path)
    contract = read_json(contract_path, "contract")
    plan = read_json(plan_path, "runtime_plan")
    require(
        plan_path.read_bytes() == canonical_bytes(plan),
        "runtime_plan: not canonical JSON",
    )
    require(isinstance(plan, dict) and set(plan) == PLAN_KEYS,
            "runtime_plan: key set mismatch")
    require(
        plan["schema"] == "s40-runtime-config-plan-v3",
        "runtime_plan: schema mismatch",
    )
    run_id = require_string(plan["run_id"], "runtime_plan.run_id")
    mode = require_string(plan["mode"], "runtime_plan.mode")
    require(mode in KINDS_BY_MODE, "runtime_plan: unsupported mode")
    event_log_path = require_string(
        plan["event_log_path"], "runtime_plan.event_log_path")
    require(Path(event_log_path).is_absolute(),
            "runtime_plan: event log path must be absolute")
    model_ids = contract["workload"]["models"]
    hot_model = require_string(
        plan["hot_model_id"], "runtime_plan.hot_model_id")
    require(hot_model in model_ids, "runtime_plan: unknown hot model")
    alternate_model = next(
        model_id for model_id in model_ids if model_id != hot_model)

    values = plan["executors"]
    require(isinstance(values, list), "runtime_plan.executors: expected array")
    executors = [_executor(value, index)
                 for index, value in enumerate(values)]
    require(
        tuple(executor["kind"] for executor in executors)
        == KINDS_BY_MODE[mode],
        "runtime_plan: executor topology mismatch",
    )
    ids = [executor["executor_id"] for executor in executors]
    require(len(set(ids)) == len(ids),
            "runtime_plan: duplicate executor ID")
    instance_ids = [
        executor["executor_instance_id"] for executor in executors]
    require(
        len(set(instance_ids)) == len(instance_ids),
        "runtime_plan: duplicate executor instance ID",
    )
    for executor in executors:
        require(
            executor["credits"] <= executor["execute_concurrency"],
            "runtime_plan: credits exceed execute concurrency",
        )
        require(
            executor["queue_capacity"] >= executor["credits"],
            "runtime_plan: queue capacity is below credits",
        )
    for executor in executors[1:]:
        require(
            executors[0]["credits"] >= executor["credits"],
            "runtime_plan: GPU credits are below warm executor credits",
        )

    profile_lock_path = plan["c3_profile_lock_path"]
    profile_lock = plan["c3_profile_lock_sha256"]
    profile_lock_record = None
    if mode == "C3_DUAL_PARTIAL_OFFLOAD":
        profile_path, profile_lock = _validate_c3_profile_lock(
            profile_lock_path, profile_lock, contract)
        profile_lock_path = str(profile_path)
        profile_lock_record = read_json(profile_path, "c3_profile_lock")
    else:
        require(profile_lock is None and profile_lock_path is None,
                "runtime_plan: unexpected C3 profile lock")
    evidence_root_path, evidence_root_sha256 = _validate_evidence_root(
        plan["evidence_root_path"],
        plan["evidence_root_sha256"],
        mode=mode,
        executor_ids=ids,
        contract_path=contract_path,
        c3_profile_lock=profile_lock_record,
    )

    output_executors = []
    for order, executor in enumerate(executors):
        output_executors.append({
            "credits": executor["credits"],
            "execute_concurrency": executor["execute_concurrency"],
            "executor_id": executor["executor_id"],
            "executor_instance_id": executor["executor_instance_id"],
            "expected_peer_pid": executor["expected_peer_pid"],
            "expected_peer_start_time_ticks":
                executor["expected_peer_start_time_ticks"],
            "order": order,
            "output_limit_bytes": executor["output_limit_bytes"],
            "queue_capacity": executor["queue_capacity"],
            "role": ROLE_BY_KIND[executor["kind"]],
            "socket_path": executor["socket_path"],
            "timeout_ms": executor["timeout_ms"],
            "transport": executor["transport"],
        })

    initial_models = []
    for executor in executors:
        kind = executor["kind"]
        if kind == "GPU_CPU_PARTIAL":
            states = {model_id: "READY" for model_id in model_ids}
        elif kind == "GPU_PRIMARY":
            states = {hot_model: "READY", alternate_model: "ABSENT"}
        else:
            states = {hot_model: "ABSENT", alternate_model: "READY"}
        for model_id in model_ids:
            initial_models.append({
                "executor_id": executor["executor_id"],
                "model_id": model_id,
                "state": states[model_id],
            })

    runtime = {
        "c3_profile_lock_sha256": profile_lock,
        "configuration": mode,
        "evidence_root_sha256": evidence_root_sha256,
        "event_log_path": event_log_path,
        "executors": output_executors,
        "initial_models": initial_models,
        "promotion_enabled": contract["matrix"][mode]["promotion_enabled"],
        "run_id": run_id,
        "runtime_plan_sha256": digest_file(plan_path),
        "schema": "llama-server-warm-tier-runtime-v4",
    }
    return {
        "c3_profile_lock_path": profile_lock_path,
        "c3_profile_lock_sha256": profile_lock,
        "contract_sha256": digest_file(contract_path),
        "evidence_root_path": str(evidence_root_path),
        "evidence_root_sha256": evidence_root_sha256,
        "mode": mode,
        "runtime": runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    try:
        result = build_runtime_config(args.plan, args.contract)
        require(args.output.is_absolute(), "output: expected absolute path")
        require(not args.output.exists(), "output: already exists")
        with args.output.open("xb", buffering=0) as sink:
            sink.write(canonical_bytes(result["runtime"]))
            sink.flush()
            os.fsync(sink.fileno())
        directory_fd = os.open(
            args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (EvidenceError, OSError) as error:
        print(f"ERROR: {error}")
        return 2
    print(canonical_bytes({
        "c3_profile_lock_path": result["c3_profile_lock_path"],
        "c3_profile_lock_sha256": result["c3_profile_lock_sha256"],
        "contract_sha256": result["contract_sha256"],
        "mode": result["mode"],
        "runtime_config_path": str(args.output),
        "runtime_config_sha256": digest_file(args.output),
        "status": "S40_RUNTIME_CONFIG_BUILT",
    }).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
