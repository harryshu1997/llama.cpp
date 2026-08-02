#!/usr/bin/python3 -I
"""Validate the V2.3 complete runtime-bundle provenance overlay."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any


sys.dont_write_bytecode = True

IDENTITY_KEYS = {
    "base_runtime_identity_sha256",
    "completed_ns",
    "phase",
    "phase_id",
    "processes",
    "runtime_bundle_fresh_sha256",
    "runtime_bundle_plan_sha256",
    "runtime_bundle_snapshot_sha256",
    "schema",
    "started_ns",
}
PROCESS_KEYS = {
    "boot_id",
    "bundle_id",
    "bundle_sha256",
    "endpoint",
    "evidence_role",
    "evidence_sha256",
    "identity_probe_sha256",
    "launcher_component_id",
    "launcher_path",
    "loaded_repo_component_ids",
    "observed_ns",
    "pid",
    "start_ticks",
    "system_dependencies",
}
SYSTEM_DEPENDENCY_KEYS = {
    "build_id",
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "path",
    "size",
}
EVIDENCE_ROLES = {
    "cuda_monolithic": "model.qwen3-14b-q4_k_m.oracle.cuda_monolithic",
    "cuda_route": "model.qwen3-14b-q4_k_m.oracle.cuda_route",
    "op12_stagenet": "model.qwen3-14b-q4_k_m.placement.op12",
    "op15_direct_relay": "model.qwen3-14b-q4_k_m.route_transfer",
    "op15_stagenet": "model.qwen3-14b-q4_k_m.placement.op15",
}
SYSTEM_ROOTS = {
    "cuda": ("/lib/", "/usr/lib/", "/usr/local/cuda/"),
    "op12": ("/apex/", "/system/", "/vendor/"),
    "op15": ("/apex/", "/system/", "/vendor/"),
}


def _read_source(path: Path, field: str) -> str:
    if not path.is_absolute():
        raise ValueError(f"E_PATH: {field}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"E_NOT_REGULAR: {field}")
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
        value.st_mode,
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise ValueError(f"E_CHANGED: {field}")
    return bytes(raw).decode("ascii")


def _load_source(
    path: Path,
    name: str,
    injected: dict[str, object] | None = None,
) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    if injected:
        module.__dict__.update(injected)
    source = _read_source(path, name)
    exec(
        compile(source, str(path), "exec", dont_inherit=True, optimize=0),
        module.__dict__,
    )
    return module


def _launcher_paths(plan: dict[str, Any]) -> tuple[str, str]:
    components = {
        value.get("component_id"): value
        for value in plan.get("components", [])
        if type(value) is dict
    }
    bundles = {
        value.get("bundle_id"): value
        for value in plan.get("bundles", [])
        if type(value) is dict
    }
    try:
        return (
            components[bundles["op15_stagenet"]["launcher_component_id"]]["path"],
            components[bundles["op12_stagenet"]["launcher_component_id"]]["path"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("E_RUNTIME_PLAN_LAUNCHERS") from error


def _role_digests(manifest: dict[str, Any], base) -> dict[str, str]:
    values = manifest.get("artifacts")
    base.require(type(values) is list and bool(values), "E_MANIFEST_ARTIFACTS")
    result = {}
    for index, value in enumerate(values):
        field = f"manifest.artifacts[{index}]"
        base.require(type(value) is dict, f"E_OBJECT: {field}")
        role = base.text(value.get("role"), f"{field}.role")
        base.require(role not in result, f"E_ROLE_REUSE: {role}")
        result[role] = base.digest(value.get("sha256"), f"{field}.sha256")
    return result


def validate_loaded_repo_components(
    value: Any,
    bundle: dict[str, Any],
    base,
    field: str,
) -> None:
    base.exact(
        value,
        bundle["required_component_ids"],
        f"E_RUNTIME_LOADED_COMPONENTS: {field}",
    )


def validate(args: argparse.Namespace) -> dict[str, Any]:
    base = _load_source(args.base_support, "s39_v23_base_overlay")
    driver = _load_source(
        args.driver_support,
        "s39_v23_runtime_bundle_driver_overlay",
        {"BASE": base},
    )
    contract, contract_raw = base.read_canonical(args.contract)
    candidate, candidate_raw = base.read_canonical(args.candidate)
    plan_unchecked, _ = base.read_canonical(args.runtime_bundle_plan)
    op15_worker, op12_worker = _launcher_paths(plan_unchecked)
    plan, plan_raw, components, bundles = driver.load_runtime_plan(
        args.runtime_bundle_plan,
        contract_raw,
        candidate_raw,
        op15_worker,
        op12_worker,
    )
    base_artifact, base_artifact_raw = base.read_canonical(
        args.base_artifact_snapshot
    )
    base_lock, base_lock_raw = base.read_canonical(args.base_readiness_lock)
    base_fresh, base_fresh_raw = base.read_canonical(args.base_fresh_snapshot)
    base_runtime, base_runtime_raw = base.read_canonical(args.base_runtime_identity)
    manifest, _ = base.read_canonical(args.manifest)
    runtime_snapshot, runtime_snapshot_raw, artifact_components = (
        driver._load_runtime_snapshot(
            args.runtime_bundle_snapshot,
            base_artifact_raw,
            plan,
            plan_raw,
            components,
            bundles,
        )
    )
    del base_artifact

    lock, lock_raw = base.read_canonical(args.runtime_bundle_readiness_lock)
    base.exact_keys(
        lock,
        {
            "event_ns",
            "phase",
            "phase_id",
            "runtime_bundle_plan_sha256",
            "runtime_bundle_snapshot_sha256",
            "schema",
        },
        "runtime_bundle_lock",
    )
    base.exact(
        lock["schema"],
        "s39-cp0-r1-runtime-bundle-readiness-lock-v1",
        "runtime_bundle_lock.schema",
    )
    base.exact(lock["phase"], manifest["phase"], "runtime_bundle_lock.phase")
    base.exact(lock["phase_id"], manifest["phase_id"], "runtime_bundle_lock.phase_id")
    base.exact(
        lock["runtime_bundle_plan_sha256"],
        base.sha256_bytes(plan_raw),
        "runtime_bundle_lock.plan",
    )
    base.exact(
        lock["runtime_bundle_snapshot_sha256"],
        base.sha256_bytes(runtime_snapshot_raw),
        "runtime_bundle_lock.snapshot",
    )
    lock_ns = base.integer(lock["event_ns"], "runtime_bundle_lock.event", 1)
    base.require(
        runtime_snapshot["completed_ns"] <= lock_ns,
        "E_RUNTIME_BUNDLE_LOCK_ORDER",
    )

    fresh, fresh_raw = base.read_canonical(args.runtime_bundle_fresh)
    base.exact_keys(
        fresh,
        {
            "base_fresh_snapshot_sha256",
            "completed_ns",
            "phase",
            "phase_id",
            "probe_intervals",
            "readiness_lock_sha256",
            "runtime_bundle_inventories",
            "runtime_bundle_plan_sha256",
            "runtime_bundle_snapshot_sha256",
            "runtime_component_stats",
            "schema",
            "started_ns",
        },
        "runtime_bundle_fresh",
    )
    base.exact(
        fresh["schema"],
        "s39-cp0-r1-runtime-bundle-fresh-v1",
        "runtime_bundle_fresh.schema",
    )
    base.exact(fresh["phase"], manifest["phase"], "runtime_bundle_fresh.phase")
    base.exact(fresh["phase_id"], manifest["phase_id"], "runtime_bundle_fresh.phase_id")
    base.exact(
        fresh["base_fresh_snapshot_sha256"],
        base.sha256_bytes(base_fresh_raw),
        "runtime_bundle_fresh.base",
    )
    base.exact(
        fresh["readiness_lock_sha256"],
        base.sha256_bytes(lock_raw),
        "runtime_bundle_fresh.lock",
    )
    base.exact(
        fresh["runtime_bundle_plan_sha256"],
        base.sha256_bytes(plan_raw),
        "runtime_bundle_fresh.plan",
    )
    base.exact(
        fresh["runtime_bundle_snapshot_sha256"],
        base.sha256_bytes(runtime_snapshot_raw),
        "runtime_bundle_fresh.snapshot",
    )
    started = base.integer(fresh["started_ns"], "runtime_bundle_fresh.started", 1)
    completed = base.integer(fresh["completed_ns"], "runtime_bundle_fresh.completed", 1)
    base.require(lock_ns <= started < completed, "E_RUNTIME_BUNDLE_FRESH_ORDER")
    base.require(
        completed < manifest["acquisition_started_ns"],
        "E_RUNTIME_BUNDLE_ACQUISITION_ORDER",
    )
    base.require(
        manifest["acquisition_started_ns"] - completed
        <= contract["readiness_v2_3"]["fresh_snapshot_maximum_age_ns"],
        "E_RUNTIME_BUNDLE_FRESH_STALE",
    )
    intervals = fresh["probe_intervals"]
    base.require(type(intervals) is list and bool(intervals), "E_PROBE_INTERVALS")
    previous_completed = lock_ns
    for index, interval in enumerate(intervals):
        field = f"runtime_bundle_fresh.probe_intervals[{index}]"
        base.exact_keys(
            interval,
            {"probe_completed_ns", "probe_index", "probe_started_ns"},
            field,
        )
        base.exact(interval["probe_index"], index, f"{field}.index")
        probe_started = base.integer(
            interval["probe_started_ns"],
            f"{field}.started",
            1,
        )
        probe_completed = base.integer(
            interval["probe_completed_ns"],
            f"{field}.completed",
            1,
        )
        base.require(
            previous_completed <= probe_started <= probe_completed <= completed,
            f"E_PROBE_INTERVAL: {field}",
        )
        previous_completed = probe_completed
    stats = fresh["runtime_component_stats"]
    base.require(
        type(stats) is list and len(stats) == len(components),
        "E_RUNTIME_COMPONENT_STATS",
    )
    for index, value in enumerate(stats):
        field = f"runtime_component_stats[{index}]"
        base.exact_keys(
            value,
            {"bundle_id", "component_id", "endpoint", "path", "stat"},
            field,
        )
        component_id = value["component_id"]
        base.require(component_id in artifact_components, f"E_COMPONENT_ID: {field}")
        expected = artifact_components[component_id]
        for key in ("bundle_id", "component_id", "endpoint", "path"):
            base.exact(value[key], expected[key], f"{field}.{key}")
        base.exact(value["stat"], expected["stat"], f"E_COMPONENT_CHANGED: {component_id}")
    base.exact(
        [value["component_id"] for value in stats],
        sorted(components),
        "runtime_component_stats.order",
    )
    base.exact(
        fresh["runtime_bundle_inventories"],
        runtime_snapshot["runtime_bundle_inventories"],
        "runtime_bundle_fresh.inventories",
    )

    identity, identity_raw = base.read_canonical(args.runtime_bundle_identity)
    base.exact_keys(identity, IDENTITY_KEYS, "runtime_bundle_identity")
    base.exact(
        identity["schema"],
        "s39-cp0-r1-runtime-bundle-runtime-identity-v1",
        "runtime_bundle_identity.schema",
    )
    base.exact(identity["phase"], manifest["phase"], "runtime_bundle_identity.phase")
    base.exact(
        identity["phase_id"],
        manifest["phase_id"],
        "runtime_bundle_identity.phase_id",
    )
    base.exact(
        identity["base_runtime_identity_sha256"],
        base.sha256_bytes(base_runtime_raw),
        "runtime_bundle_identity.base",
    )
    base.exact(
        identity["runtime_bundle_plan_sha256"],
        base.sha256_bytes(plan_raw),
        "runtime_bundle_identity.plan",
    )
    base.exact(
        identity["runtime_bundle_snapshot_sha256"],
        base.sha256_bytes(runtime_snapshot_raw),
        "runtime_bundle_identity.snapshot",
    )
    base.exact(
        identity["runtime_bundle_fresh_sha256"],
        base.sha256_bytes(fresh_raw),
        "runtime_bundle_identity.fresh",
    )
    identity_started = base.integer(
        identity["started_ns"],
        "runtime_bundle_identity.started",
        1,
    )
    identity_completed = base.integer(
        identity["completed_ns"],
        "runtime_bundle_identity.completed",
        1,
    )
    base.require(
        manifest["acquisition_started_ns"]
        <= identity_started
        < identity_completed
        <= manifest["phase_closed_ns"],
        "E_RUNTIME_BUNDLE_IDENTITY_INTERVAL",
    )

    role_digests = _role_digests(manifest, base)
    base_executors = {
        value["executor_id"]: value
        for value in base_runtime["executors"]
        if type(value) is dict and type(value.get("executor_id")) is str
    }
    base.require(
        set(base_executors) == {"GPU", "PHONE_OP12", "PHONE_OP15"},
        "E_BASE_EXECUTORS",
    )
    boot_ids = {
        "cuda": base_executors["GPU"]["host_boot_id"],
        "op12": base_executors["PHONE_OP12"]["boot_id"],
        "op15": base_executors["PHONE_OP15"]["boot_id"],
    }
    processes = identity["processes"]
    base.require(
        type(processes) is list and len(processes) == len(bundles),
        "E_RUNTIME_BUNDLE_PROCESSES",
    )
    for index, value in enumerate(processes):
        field = f"runtime_bundle_identity.processes[{index}]"
        base.exact_keys(value, PROCESS_KEYS, field)
        bundle_id = value["bundle_id"]
        base.require(bundle_id in bundles, f"E_RUNTIME_BUNDLE_PROCESS: {field}")
        bundle = bundles[bundle_id]
        bundle_snapshot = next(
            item
            for item in runtime_snapshot["runtime_bundles"]
            if item["bundle_id"] == bundle_id
        )
        for key in ("bundle_id", "endpoint", "launcher_component_id"):
            base.exact(value[key], bundle[key], f"{field}.{key}")
        base.exact(
            value["bundle_sha256"],
            bundle_snapshot["bundle_sha256"],
            f"{field}.bundle_sha256",
        )
        evidence_role = EVIDENCE_ROLES[bundle_id]
        base.exact(value["evidence_role"], evidence_role, f"{field}.evidence_role")
        base.require(evidence_role in role_digests, f"E_EVIDENCE_ROLE: {bundle_id}")
        base.exact(
            value["evidence_sha256"],
            role_digests[evidence_role],
            f"{field}.evidence_sha256",
        )
        base.exact(value["boot_id"], boot_ids[bundle["endpoint"]], f"{field}.boot")
        base.digest(
            value["identity_probe_sha256"],
            f"{field}.identity_probe_sha256",
        )
        base.integer(value["pid"], f"{field}.pid", 1)
        base.integer(value["start_ticks"], f"{field}.start_ticks", 1)
        observed_ns = base.integer(value["observed_ns"], f"{field}.observed_ns", 1)
        base.require(
            identity_started <= observed_ns <= identity_completed,
            f"E_RUNTIME_OBSERVED: {bundle_id}",
        )
        launcher = components[bundle["launcher_component_id"]]
        base.exact(value["launcher_path"], launcher["path"], f"{field}.launcher_path")
        validate_loaded_repo_components(
            value["loaded_repo_component_ids"],
            bundle,
            base,
            bundle_id,
        )
        if bundle_id in ("op15_stagenet", "op12_stagenet"):
            executor = base_executors[
                "PHONE_OP15" if bundle["endpoint"] == "op15" else "PHONE_OP12"
            ]
            base.exact(value["pid"], executor["worker_pid"], f"{field}.worker_pid")
            base.exact(
                value["start_ticks"],
                executor["worker_start_ticks"],
                f"{field}.worker_start_ticks",
            )
            base.exact(
                value["launcher_path"],
                executor["worker_executable_path"],
                f"{field}.worker_path",
            )
        dependencies = value["system_dependencies"]
        base.require(type(dependencies) is list and bool(dependencies), f"E_SYSTEM_DEPS: {field}")
        previous_path = None
        for dep_index, dependency in enumerate(dependencies):
            dep_field = f"{field}.system_dependencies[{dep_index}]"
            base.exact_keys(dependency, SYSTEM_DEPENDENCY_KEYS, dep_field)
            path = base.text(dependency["path"], f"{dep_field}.path")
            base.require(
                any(path.startswith(root) for root in SYSTEM_ROOTS[bundle["endpoint"]]),
                f"E_SYSTEM_DEP_PATH: {path}",
            )
            if previous_path is not None:
                base.require(previous_path < path, f"E_SYSTEM_DEP_ORDER: {field}")
            previous_path = path
            for key in (
                "ctime_ns",
                "device_id",
                "inode",
                "mode",
                "mtime_ns",
                "size",
            ):
                base.integer(dependency[key], f"{dep_field}.{key}")
            build_id = dependency["build_id"]
            base.require(
                build_id is None or (type(build_id) is str and bool(build_id)),
                f"E_SYSTEM_BUILD_ID: {dep_field}",
            )
    base.exact(
        [value["bundle_id"] for value in processes],
        sorted(bundles),
        "runtime_bundle_identity.process_order",
    )
    return {
        "runtime_bundle_fresh_sha256": base.sha256_bytes(fresh_raw),
        "runtime_bundle_identity_sha256": base.sha256_bytes(identity_raw),
        "runtime_bundle_plan_sha256": base.sha256_bytes(plan_raw),
        "runtime_bundle_snapshot_sha256": base.sha256_bytes(runtime_snapshot_raw),
        "schema": "s39-cp0-r1-runtime-bundle-validation-v1",
        "status": "RUNTIME_BUNDLE_PROVENANCE_PASS",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "base-support",
        "driver-support",
        "contract",
        "candidate",
        "runtime-bundle-plan",
        "manifest",
        "base-artifact-snapshot",
        "base-readiness-lock",
        "base-fresh-snapshot",
        "base-runtime-identity",
        "runtime-bundle-snapshot",
        "runtime-bundle-readiness-lock",
        "runtime-bundle-fresh",
        "runtime-bundle-identity",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = validate(parse_args(argv))
    except Exception as error:
        result = {
            "error": f"{type(error).__name__}: {error}",
            "schema": "s39-cp0-r1-runtime-bundle-validation-v1",
            "status": "RUNTIME_BUNDLE_PROVENANCE_REFUSED",
        }
        print(
            __import__("json").dumps(
                result,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(
        __import__("json").dumps(
            result,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
