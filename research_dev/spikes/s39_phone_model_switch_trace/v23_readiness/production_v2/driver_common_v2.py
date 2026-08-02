#!/usr/bin/env python3
"""Runtime-bundle extension for the CP0-R1 V2.3 readiness producers."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys
from typing import Any, Callable


sys.dont_write_bytecode = True

try:
    BASE
except NameError as error:
    raise RuntimeError("driver_common_v2.py requires injected BASE support") from error


RUNTIME_PLAN_KEYS = {
    "bundle_roots",
    "bundles",
    "candidate_sha256",
    "components",
    "contract_sha256",
    "model_id",
    "phase",
    "schema",
}
COMPONENT_KEYS = {
    "bundle_id",
    "bytes",
    "component_id",
    "endpoint",
    "path",
    "role",
    "sha256",
}
BUNDLE_KEYS = {
    "bundle_id",
    "endpoint",
    "launcher_component_id",
    "process_role",
    "required_component_ids",
}
REQUIRED_BUNDLES = {
    "cuda_monolithic": ("cuda", "cuda_monolithic"),
    "cuda_route": ("cuda", "cuda_route"),
    "op12_stagenet": ("op12", "stagenet_worker"),
    "op15_direct_relay": ("op15", "direct_relay"),
    "op15_stagenet": ("op15", "stagenet_worker"),
}
ENDPOINTS = {"cuda", "op12", "op15"}
COMPONENT_ROLES = {"backend_library", "executable", "shared_library"}


class ProbeIntervalRunner:
    def __init__(self, runner, now_ns: Callable[[], int]):
        self.runner = runner
        self.now_ns = now_ns
        self.intervals = []

    def run(self, argv: list[str], *, timeout: int):
        started_ns = self.now_ns()
        result = self.runner.run(argv, timeout=timeout)
        completed_ns = self.now_ns()
        BASE.require(started_ns <= completed_ns, "E_PROBE_INTERVAL")
        self.intervals.append({
            "probe_completed_ns": completed_ns,
            "probe_index": len(self.intervals),
            "probe_started_ns": started_ns,
        })
        return result


def _path_under_root(path: str, root: str, field: str) -> None:
    path_value = Path(path)
    root_value = Path(root)
    BASE.require(
        path_value.is_absolute()
        and root_value.is_absolute()
        and ".." not in path_value.parts
        and ".." not in root_value.parts
        and path_value.is_relative_to(root_value),
        f"E_RUNTIME_ROOT: {field}",
    )


def load_runtime_plan(
    path: Path,
    contract_raw: bytes,
    candidate_raw: bytes,
    op15_worker: str,
    op12_worker: str,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    plan, raw = BASE.read_canonical(path)
    BASE.exact_keys(plan, RUNTIME_PLAN_KEYS, "runtime_bundle_plan")
    BASE.exact(
        plan["schema"],
        "s39-cp0-r1-runtime-bundle-plan-v1",
        "runtime_bundle_plan.schema",
    )
    BASE.exact(plan["phase"], BASE.PHASE, "runtime_bundle_plan.phase")
    BASE.exact(plan["model_id"], BASE.MODEL_ID, "runtime_bundle_plan.model")
    BASE.exact(
        plan["contract_sha256"],
        BASE.sha256_bytes(contract_raw),
        "runtime_bundle_plan.contract",
    )
    BASE.exact(
        plan["candidate_sha256"],
        BASE.sha256_bytes(candidate_raw),
        "runtime_bundle_plan.candidate",
    )

    roots = BASE.exact_keys(
        plan["bundle_roots"],
        set(REQUIRED_BUNDLES),
        "runtime_bundle_plan.bundle_roots",
    )
    normalized_roots = {}
    for bundle_id in sorted(roots):
        BASE.validate_absolute_path(
            roots[bundle_id],
            f"runtime_bundle_plan.bundle_roots.{bundle_id}",
        )
        BASE.require(
            ".." not in Path(roots[bundle_id]).parts,
            f"E_RUNTIME_ROOT: {bundle_id}",
        )
        normalized_roots[bundle_id] = Path(roots[bundle_id])
    for left_id, left in normalized_roots.items():
        for right_id, right in normalized_roots.items():
            if left_id != right_id:
                BASE.require(
                    not left.is_relative_to(right),
                    f"E_RUNTIME_ROOT_OVERLAP: {left_id}:{right_id}",
                )

    values = plan["components"]
    BASE.require(type(values) is list and bool(values), "E_RUNTIME_COMPONENTS")
    components: dict[str, dict[str, Any]] = {}
    locations = set()
    previous_id = None
    for index, value in enumerate(values):
        field = f"runtime_bundle_plan.components[{index}]"
        value = BASE.exact_keys(value, COMPONENT_KEYS, field)
        component_id = BASE.text(value["component_id"], f"{field}.component_id")
        BASE.require(
            len(component_id) <= 128
            and all(character.isalnum() or character in "._-" for character in component_id),
            f"E_COMPONENT_ID: {field}",
        )
        BASE.require(component_id not in components, f"E_COMPONENT_REUSE: {component_id}")
        if previous_id is not None:
            BASE.require(previous_id < component_id, "E_COMPONENT_ORDER")
        previous_id = component_id
        endpoint = BASE.text(value["endpoint"], f"{field}.endpoint")
        BASE.require(endpoint in ENDPOINTS, f"E_COMPONENT_ENDPOINT: {field}")
        bundle_id = BASE.text(value["bundle_id"], f"{field}.bundle_id")
        BASE.require(bundle_id in REQUIRED_BUNDLES, f"E_COMPONENT_BUNDLE: {field}")
        BASE.exact(
            endpoint,
            REQUIRED_BUNDLES[bundle_id][0],
            f"E_COMPONENT_BUNDLE_ENDPOINT: {field}",
        )
        role = BASE.text(value["role"], f"{field}.role")
        BASE.require(role in COMPONENT_ROLES, f"E_COMPONENT_ROLE: {field}")
        component_path = BASE.validate_absolute_path(value["path"], f"{field}.path")
        _path_under_root(component_path, roots[bundle_id], field)
        location = (endpoint, component_path)
        BASE.require(location not in locations, f"E_COMPONENT_PATH_REUSE: {field}")
        locations.add(location)
        BASE.integer(value["bytes"], f"{field}.bytes", 1)
        BASE.digest(value["sha256"], f"{field}.sha256")
        components[component_id] = value

    values = plan["bundles"]
    BASE.require(
        type(values) is list and len(values) == len(REQUIRED_BUNDLES),
        "E_RUNTIME_BUNDLES",
    )
    bundles: dict[str, dict[str, Any]] = {}
    previous_id = None
    referenced = set()
    for index, value in enumerate(values):
        field = f"runtime_bundle_plan.bundles[{index}]"
        value = BASE.exact_keys(value, BUNDLE_KEYS, field)
        bundle_id = BASE.text(value["bundle_id"], f"{field}.bundle_id")
        BASE.require(bundle_id in REQUIRED_BUNDLES, f"E_BUNDLE_ID: {field}")
        BASE.require(bundle_id not in bundles, f"E_BUNDLE_REUSE: {bundle_id}")
        if previous_id is not None:
            BASE.require(previous_id < bundle_id, "E_BUNDLE_ORDER")
        previous_id = bundle_id
        endpoint, process_role = REQUIRED_BUNDLES[bundle_id]
        BASE.exact(value["endpoint"], endpoint, f"{field}.endpoint")
        BASE.exact(value["process_role"], process_role, f"{field}.process_role")
        required = value["required_component_ids"]
        BASE.require(
            type(required) is list
            and len(required) >= 2
            and all(type(item) is str for item in required)
            and required == sorted(set(required)),
            f"E_BUNDLE_COMPONENTS: {field}",
        )
        BASE.require(
            any(
                components[component_id]["role"]
                in ("backend_library", "shared_library")
                for component_id in required
                if component_id in components
            ),
            f"E_BUNDLE_RUNTIME_LIBRARY: {field}",
        )
        launcher = BASE.text(
            value["launcher_component_id"],
            f"{field}.launcher_component_id",
        )
        BASE.require(launcher in required, f"E_BUNDLE_LAUNCHER: {field}")
        for component_id in required:
            BASE.require(component_id in components, f"E_BUNDLE_COMPONENT: {component_id}")
            BASE.exact(
                components[component_id]["endpoint"],
                endpoint,
                f"E_BUNDLE_COMPONENT_ENDPOINT: {component_id}",
            )
            BASE.exact(
                components[component_id]["bundle_id"],
                bundle_id,
                f"E_BUNDLE_COMPONENT_OWNER: {component_id}",
            )
        BASE.exact(
            components[launcher]["role"],
            "executable",
            f"E_BUNDLE_LAUNCHER_ROLE: {field}",
        )
        referenced.update(required)
        bundles[bundle_id] = value
    BASE.exact(set(bundles), set(REQUIRED_BUNDLES), "runtime_bundle_plan.bundle_ids")
    BASE.exact(referenced, set(components), "runtime_bundle_plan.unused_components")
    BASE.exact(
        components[bundles["op15_stagenet"]["launcher_component_id"]]["path"],
        op15_worker,
        "runtime_bundle_plan.op15_worker",
    )
    BASE.exact(
        components[bundles["op12_stagenet"]["launcher_component_id"]]["path"],
        op12_worker,
        "runtime_bundle_plan.op12_worker",
    )
    return plan, raw, components, bundles


def bundle_identity(
    bundle: dict[str, Any],
    components: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "bundle_id": bundle["bundle_id"],
        "components": [
            {
                "bundle_id": components[component_id]["bundle_id"],
                "bytes": components[component_id]["bytes"],
                "component_id": component_id,
                "path": components[component_id]["path"],
                "role": components[component_id]["role"],
                "sha256": components[component_id]["sha256"],
            }
            for component_id in bundle["required_component_ids"]
        ],
        "endpoint": bundle["endpoint"],
        "launcher_component_id": bundle["launcher_component_id"],
        "process_role": bundle["process_role"],
        "schema": "s39-cp0-r1-runtime-bundle-identity-v1",
    }


def bundle_record(
    bundle: dict[str, Any],
    components: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    identity = bundle_identity(bundle, components)
    return {
        "bundle_id": bundle["bundle_id"],
        "bundle_sha256": BASE.sha256_bytes(BASE.canonical_bytes(identity)),
        "endpoint": bundle["endpoint"],
        "launcher_component_id": bundle["launcher_component_id"],
        "process_role": bundle["process_role"],
        "required_component_ids": bundle["required_component_ids"],
    }


def _spec_for_component(
    contract: dict[str, Any],
    component: dict[str, Any],
) -> dict[str, Any]:
    spec = {
        "bytes": component["bytes"],
        "digest": component["sha256"],
        "path": component["path"],
    }
    if component["endpoint"] in ("op15", "op12"):
        spec["adb_selector"] = contract["readiness_v2_3"][
            "phone_identity"
        ][component["endpoint"]]["adb_selector"]
    return spec


def _collect_component(
    runner,
    contract: dict[str, Any],
    component: dict[str, Any],
    timeout: int,
    include_digest: bool,
) -> dict[str, Any]:
    spec = _spec_for_component(contract, component)
    collected = BASE.collect_artifact(
        runner,
        contract["preflight"]["ssh_target"],
        contract["preflight"]["phone_adb_port"],
        component["endpoint"],
        spec,
        timeout,
        include_digest,
    )
    if include_digest:
        BASE.validate_collected_artifact(
            component["component_id"],
            spec,
            collected,
        )
    return collected


REMOTE_INVENTORY_PYTHON = """\
import json,os,stat,sys
root=sys.argv[1]
if not os.path.isabs(root) or os.path.islink(root) or not os.path.isdir(root):
 raise SystemExit(40)
files=[]
bad=[]
stack=[root]
while stack:
 current=stack.pop()
 with os.scandir(current) as entries:
  for entry in entries:
   path=entry.path
   info=entry.stat(follow_symlinks=False)
   if stat.S_ISDIR(info.st_mode):
    stack.append(path)
   elif stat.S_ISREG(info.st_mode):
    files.append(path)
   else:
    bad.append(path)
print(json.dumps({"bad":sorted(bad),"files":sorted(files)},sort_keys=True,separators=(",",":")))
"""


def _android_inventory_command(root: str) -> str:
    quoted = shlex.quote(root)
    return (
        "set -eu; "
        f"test -d {quoted}; test ! -L {quoted}; "
        f"find {quoted} -mindepth 1 -print | sort | "
        "while IFS= read -r path; do "
        "if test -L \"$path\"; then printf 'BAD=%s\\n' \"$path\"; "
        "elif test -d \"$path\"; then :; "
        "elif test -f \"$path\"; then printf 'FILE=%s\\n' \"$path\"; "
        "else printf 'BAD=%s\\n' \"$path\"; fi; done"
    )


def _collect_inventory(
    runner,
    contract: dict[str, Any],
    endpoint: str,
    root: str,
    timeout: int,
) -> list[str]:
    if endpoint == "cuda":
        raw = BASE.run_probe(
            runner,
            BASE.ssh_python_argv(
                contract["preflight"]["ssh_target"],
                REMOTE_INVENTORY_PYTHON,
                root,
            ),
            timeout,
            f"inventory.{endpoint}",
        )
        value = BASE.parse_json(raw, f"inventory.{endpoint}")
        BASE.exact_keys(value, {"bad", "files"}, f"inventory.{endpoint}")
        BASE.exact(value["bad"], [], f"E_RUNTIME_NONREGULAR: {endpoint}")
        files = value["files"]
        BASE.require(
            type(files) is list
            and all(type(path) is str for path in files)
            and files == sorted(set(files)),
            f"E_RUNTIME_INVENTORY: {endpoint}",
        )
        return files
    raw = BASE.run_probe(
        runner,
        BASE.adb_argv(
            contract["preflight"]["phone_adb_port"],
            contract["devices"][endpoint]["serial"],
            _android_inventory_command(root),
        ),
        timeout,
        f"inventory.{endpoint}",
    )
    files = []
    bad = []
    for line in raw.decode("ascii").splitlines():
        if line.startswith("FILE="):
            files.append(line[5:])
        elif line.startswith("BAD="):
            bad.append(line[4:])
        else:
            raise BASE.DriverError(f"E_RUNTIME_INVENTORY_LINE: {endpoint}")
    BASE.exact(bad, [], f"E_RUNTIME_NONREGULAR: {endpoint}")
    BASE.require(
        files == sorted(set(files)),
        f"E_RUNTIME_INVENTORY: {endpoint}",
    )
    return files


def _inventory_records(
    runner,
    contract: dict[str, Any],
    plan: dict[str, Any],
    components: dict[str, dict[str, Any]],
    bundles: dict[str, dict[str, Any]],
    timeout: int,
) -> list[dict[str, Any]]:
    result = []
    for bundle_id in sorted(bundles):
        bundle = bundles[bundle_id]
        expected_files = sorted(
            components[component_id]["path"]
            for component_id in bundle["required_component_ids"]
        )
        files = _collect_inventory(
            runner,
            contract,
            bundle["endpoint"],
            plan["bundle_roots"][bundle_id],
            timeout,
        )
        BASE.exact(files, expected_files, f"E_RUNTIME_DIRECTORY_SET: {bundle_id}")
        result.append({
            "bundle_id": bundle_id,
            "endpoint": bundle["endpoint"],
            "files": files,
            "root": plan["bundle_roots"][bundle_id],
        })
    return result


def artifact_driver(
    *,
    contract_path: Path,
    candidate_path: Path,
    runtime_bundle_plan_path: Path,
    pre_dir: Path,
    output_dir: Path,
    phase_id: str,
    op15_worker: str,
    op12_worker: str,
    timeout: int,
    runner=None,
    now_ns: Callable[[], int] = BASE.clock_ns,
) -> dict[str, Any]:
    base_snapshot = BASE.artifact_driver(
        contract_path=contract_path,
        candidate_path=candidate_path,
        pre_dir=pre_dir,
        output_dir=output_dir,
        phase_id=phase_id,
        op15_worker=op15_worker,
        op12_worker=op12_worker,
        timeout=timeout,
        runner=runner,
        now_ns=now_ns,
    )
    contract, _, _, _, _ = BASE.load_inputs(
        contract_path,
        candidate_path,
        pre_dir,
        phase_id,
    )
    _, contract_raw = BASE.read_canonical(contract_path)
    _, candidate_raw = BASE.read_canonical(candidate_path)
    plan, plan_raw, components, bundles = load_runtime_plan(
        runtime_bundle_plan_path,
        contract_raw,
        candidate_raw,
        op15_worker,
        op12_worker,
    )
    runner = runner or BASE.SubprocessProbeRunner()
    timeout = BASE.integer(timeout, "timeout", 1)
    BASE.require(timeout <= 7200, "E_TIMEOUT_RANGE")
    started_ns = now_ns()

    component_records = []
    for component_id in sorted(components):
        component = components[component_id]
        collected = _collect_component(runner, contract, component, timeout, True)
        component_records.append({
            "bundle_id": component["bundle_id"],
            "bytes": collected["stat"]["size"],
            "component_id": component_id,
            "endpoint": component["endpoint"],
            "path": component["path"],
            "role": component["role"],
            "sha256": collected["sha256"],
            "stat": collected["stat"],
        })
    inventories = _inventory_records(
        runner,
        contract,
        plan,
        components,
        bundles,
        timeout,
    )
    completed_ns = now_ns()
    BASE.require(started_ns < completed_ns, "E_ARTIFACT_INTERVAL")
    result = {
        "base_artifact_snapshot_sha256": BASE.sha256_bytes(
            BASE.canonical_bytes(base_snapshot)
        ),
        "completed_ns": completed_ns,
        "model_id": BASE.MODEL_ID,
        "phase": BASE.PHASE,
        "runtime_bundle_plan_sha256": BASE.sha256_bytes(plan_raw),
        "runtime_bundle_inventories": inventories,
        "runtime_bundles": [
            bundle_record(bundles[bundle_id], components)
            for bundle_id in sorted(bundles)
        ],
        "runtime_components": component_records,
        "schema": "s39-cp0-r1-runtime-bundle-snapshot-v1",
        "slot": "A",
        "started_ns": started_ns,
    }
    BASE.durable_json(output_dir / "runtime_bundle_snapshot.json", result)
    return result


def _load_runtime_snapshot(
    path: Path,
    base_artifact_raw: bytes,
    plan: dict[str, Any],
    plan_raw: bytes,
    components: dict[str, dict[str, Any]],
    bundles: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    snapshot, raw = BASE.read_canonical(path)
    BASE.exact_keys(
        snapshot,
        {
            "base_artifact_snapshot_sha256",
            "completed_ns",
            "model_id",
            "phase",
            "runtime_bundle_plan_sha256",
            "runtime_bundle_inventories",
            "runtime_bundles",
            "runtime_components",
            "schema",
            "slot",
            "started_ns",
        },
        "runtime_bundle_snapshot",
    )
    BASE.exact(
        snapshot.get("schema"),
        "s39-cp0-r1-runtime-bundle-snapshot-v1",
        "runtime_bundle_snapshot.schema",
    )
    BASE.exact(snapshot.get("phase"), BASE.PHASE, "runtime_bundle_snapshot.phase")
    BASE.exact(snapshot.get("model_id"), BASE.MODEL_ID, "runtime_bundle_snapshot.model")
    BASE.exact(snapshot.get("slot"), "A", "runtime_bundle_snapshot.slot")
    BASE.exact(
        snapshot.get("base_artifact_snapshot_sha256"),
        BASE.sha256_bytes(base_artifact_raw),
        "runtime_bundle_snapshot.base",
    )
    BASE.exact(
        snapshot.get("runtime_bundle_plan_sha256"),
        BASE.sha256_bytes(plan_raw),
        "runtime_bundle_snapshot.plan",
    )
    started_ns = BASE.integer(snapshot["started_ns"], "runtime_snapshot.started", 1)
    completed_ns = BASE.integer(snapshot["completed_ns"], "runtime_snapshot.completed", 1)
    BASE.require(started_ns < completed_ns, "E_RUNTIME_SNAPSHOT_INTERVAL")

    records = snapshot.get("runtime_components")
    BASE.require(
        type(records) is list and len(records) == len(components),
        "E_RUNTIME_COMPONENT_COUNT",
    )
    by_component = {}
    for index, record in enumerate(records):
        field = f"runtime_components[{index}]"
        BASE.exact_keys(
            record,
            {
                "bytes",
                "bundle_id",
                "component_id",
                "endpoint",
                "path",
                "role",
                "sha256",
                "stat",
            },
            field,
        )
        component_id = record["component_id"]
        BASE.require(
            component_id in components and component_id not in by_component,
            f"E_RUNTIME_COMPONENT: {field}",
        )
        expected = components[component_id]
        for key in ("bundle_id", "bytes", "endpoint", "path", "role", "sha256"):
            BASE.exact(record[key], expected[key], f"{field}.{key}")
        BASE.validate_stat(record["stat"], f"{field}.stat")
        BASE.exact(record["bytes"], record["stat"]["size"], f"{field}.size")
        by_component[component_id] = record
    BASE.exact(tuple(by_component), tuple(sorted(components)), "runtime_component.order")
    expected_bundles = [
        bundle_record(bundles[bundle_id], components)
        for bundle_id in sorted(bundles)
    ]
    BASE.exact(snapshot.get("runtime_bundles"), expected_bundles, "runtime_bundles")
    expected_inventories = [
        {
            "bundle_id": bundle_id,
            "endpoint": bundles[bundle_id]["endpoint"],
            "files": sorted(
                components[component_id]["path"]
                for component_id in bundles[bundle_id]["required_component_ids"]
            ),
            "root": plan["bundle_roots"][bundle_id],
        }
        for bundle_id in sorted(bundles)
    ]
    BASE.exact(
        snapshot.get("runtime_bundle_inventories"),
        expected_inventories,
        "runtime_bundle_inventories",
    )
    return snapshot, raw, by_component


def fresh_driver(
    *,
    contract_path: Path,
    candidate_path: Path,
    runtime_bundle_plan_path: Path,
    pre_dir: Path,
    output_dir: Path,
    phase_id: str,
    op15_worker: str,
    op12_worker: str,
    timeout: int,
    runner=None,
    now_ns: Callable[[], int] = BASE.clock_ns,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract, _, _, _, _ = BASE.load_inputs(
        contract_path,
        candidate_path,
        pre_dir,
        phase_id,
    )
    _, contract_raw = BASE.read_canonical(contract_path)
    _, candidate_raw = BASE.read_canonical(candidate_path)
    plan, plan_raw, components, bundles = load_runtime_plan(
        runtime_bundle_plan_path,
        contract_raw,
        candidate_raw,
        op15_worker,
        op12_worker,
    )
    base_artifact_path = pre_dir.parent / "artifact" / "artifact_snapshot.json"
    _, base_artifact_raw = BASE.read_canonical(base_artifact_path)
    runtime_snapshot_path = (
        pre_dir.parent / "artifact" / "runtime_bundle_snapshot.json"
    )
    runtime_snapshot, runtime_snapshot_raw, artifact_components = (
        _load_runtime_snapshot(
            runtime_snapshot_path,
            base_artifact_raw,
            plan,
            plan_raw,
            components,
            bundles,
        )
    )
    runner = runner or BASE.SubprocessProbeRunner()
    interval_runner = ProbeIntervalRunner(runner, now_ns)
    timeout = BASE.integer(timeout, "timeout", 1)
    BASE.require(timeout <= 7200, "E_TIMEOUT_RANGE")

    lock_event_ns = now_ns()
    BASE.require(
        runtime_snapshot["completed_ns"] <= lock_event_ns,
        "E_READINESS_LOCK_BEFORE_ARTIFACT",
    )
    lock = {
        "event_ns": lock_event_ns,
        "phase": BASE.PHASE,
        "phase_id": phase_id,
        "runtime_bundle_plan_sha256": BASE.sha256_bytes(plan_raw),
        "runtime_bundle_snapshot_sha256": BASE.sha256_bytes(
            runtime_snapshot_raw
        ),
        "schema": "s39-cp0-r1-runtime-bundle-readiness-lock-v1",
    }
    lock_raw = BASE.durable_json(
        output_dir / "runtime_bundle_readiness_lock.json",
        lock,
    )
    started_ns = now_ns()
    BASE.require(lock_event_ns <= started_ns, "E_FRESH_PRELOCK")
    base_lock, base_fresh = BASE.fresh_driver(
        contract_path=contract_path,
        candidate_path=candidate_path,
        pre_dir=pre_dir,
        output_dir=output_dir,
        phase_id=phase_id,
        op15_worker=op15_worker,
        op12_worker=op12_worker,
        timeout=timeout,
        runner=interval_runner,
        now_ns=now_ns,
    )

    component_stats = []
    for component_id in sorted(components):
        component = components[component_id]
        collected = _collect_component(
            interval_runner,
            contract,
            component,
            timeout,
            False,
        )
        BASE.exact(
            collected["stat"],
            artifact_components[component_id]["stat"],
            f"E_RUNTIME_COMPONENT_CHANGED: {component_id}",
        )
        component_stats.append({
            "bundle_id": component["bundle_id"],
            "component_id": component_id,
            "endpoint": component["endpoint"],
            "path": component["path"],
            "stat": collected["stat"],
        })
    inventories = _inventory_records(
        interval_runner,
        contract,
        plan,
        components,
        bundles,
        timeout,
    )

    completed_ns = now_ns()
    BASE.require(started_ns < completed_ns, "E_FRESH_INTERVAL")
    maximum_age_ns = contract["readiness_v2_3"]["fresh_snapshot_maximum_age_ns"]
    BASE.require(
        completed_ns - lock_event_ns <= maximum_age_ns,
        "E_FRESH_INTERVAL_MAXIMUM",
    )
    for interval in interval_runner.intervals:
        BASE.require(
            interval["probe_completed_ns"] - interval["probe_started_ns"]
            <= maximum_age_ns,
            "E_PROBE_INTERVAL_MAXIMUM",
        )
        BASE.require(
            completed_ns - interval["probe_completed_ns"] <= maximum_age_ns,
            "E_PROBE_STALE",
        )
    fresh = {
        "base_fresh_snapshot_sha256": BASE.sha256_bytes(
            BASE.canonical_bytes(base_fresh)
        ),
        "completed_ns": completed_ns,
        "phase": BASE.PHASE,
        "phase_id": phase_id,
        "probe_intervals": interval_runner.intervals,
        "readiness_lock_sha256": BASE.sha256_bytes(lock_raw),
        "runtime_bundle_plan_sha256": BASE.sha256_bytes(plan_raw),
        "runtime_bundle_inventories": inventories,
        "runtime_component_stats": component_stats,
        "runtime_bundle_snapshot_sha256": BASE.sha256_bytes(
            runtime_snapshot_raw
        ),
        "schema": "s39-cp0-r1-runtime-bundle-fresh-v1",
        "started_ns": started_ns,
    }
    BASE.durable_json(output_dir / "runtime_bundle_fresh.json", fresh)
    return lock, fresh


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    BASE.add_common_arguments(parser)
    parser.add_argument("--runtime-bundle-plan", type=Path, required=True)


def artifact_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture V2.3 runtime-bundle artifacts")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    try:
        value = artifact_driver(
            contract_path=args.contract,
            candidate_path=args.candidate,
            runtime_bundle_plan_path=args.runtime_bundle_plan,
            pre_dir=args.pre,
            output_dir=args.output,
            phase_id=args.phase_id,
            op15_worker=args.op15_worker,
            op12_worker=args.op12_worker,
            timeout=args.timeout_seconds,
        )
        print(
            "V23_RUNTIME_BUNDLE_ARTIFACT_PASS "
            + BASE.sha256_bytes(BASE.canonical_bytes(value))
        )
        return 0
    except Exception as error:
        print(
            f"V23_RUNTIME_BUNDLE_ARTIFACT_REFUSED: "
            f"{type(error).__name__}: {error}"
        )
        return 2


def fresh_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture V2.3 runtime-bundle readiness")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    try:
        lock, fresh = fresh_driver(
            contract_path=args.contract,
            candidate_path=args.candidate,
            runtime_bundle_plan_path=args.runtime_bundle_plan,
            pre_dir=args.pre,
            output_dir=args.output,
            phase_id=args.phase_id,
            op15_worker=args.op15_worker,
            op12_worker=args.op12_worker,
            timeout=args.timeout_seconds,
        )
        print(
            "V23_RUNTIME_BUNDLE_FRESH_PASS "
            + BASE.sha256_bytes(BASE.canonical_bytes(lock))
            + " "
            + BASE.sha256_bytes(BASE.canonical_bytes(fresh))
        )
        return 0
    except Exception as error:
        print(
            f"V23_RUNTIME_BUNDLE_FRESH_REFUSED: "
            f"{type(error).__name__}: {error}"
        )
        return 2
