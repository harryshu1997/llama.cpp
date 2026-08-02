#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
if str(S39) not in sys.path:
    sys.path.insert(0, str(S39))

import cp0_r1_evidence_v2 as v2
import cp0_r1_evidence_v21 as v21
import cp0_r1_evidence_v22 as v22

import build_contract_v23 as builder
import v23_common as common


DEFAULT_CONTRACT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_3.json"
DEFAULT_CANDIDATE = S39 / "CP0_R1_CANDIDATE.json"


def validate_inputs(
    contract_path: Path,
    candidate_path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    contract, contract_raw = common.read_canonical(contract_path)
    common.exact(contract, builder.build_contract(), "contract")
    candidate, candidate_raw = common.read_canonical(candidate_path)
    common.exact(
        common.sha256_bytes(candidate_raw),
        builder.CANDIDATE_SHA256,
        "candidate.sha256",
    )
    return contract, contract_raw, candidate, candidate_raw


def load_v22_manifest(root: Path) -> tuple[dict[str, Any], bytes]:
    raw = v2.secure_read(root, v22.MANIFEST_NAME)
    manifest = v2.parse_json(raw, v22.MANIFEST_NAME)
    v2.require(type(manifest) is dict, "E_TYPE: V2.2 manifest")
    v2.require(v2.canonical_bytes(manifest) == raw, "E_CANONICAL: V2.2 manifest")
    return manifest, raw


def reevaluate_v22_bundle(
    root: Path,
    candidate_path: Path,
    a_bundle_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    (
        contract,
        contract_raw,
        candidate,
        candidate_raw,
        parent,
        frozen_corpus,
    ) = v22.validate_inputs(v22.DEFAULT_CONTRACT, candidate_path)
    manifest, manifest_raw = load_v22_manifest(root)
    prior_results = []
    if manifest["phase"] == "B_ONLY":
        common.require(a_bundle_root is not None, "E_V22_PHASE_CHAIN: missing A")
        prior_results.append(
            v22.evaluate_root(
                a_bundle_root,
                v22.MANIFEST_NAME,
                contract,
                contract_raw,
                candidate,
                candidate_raw,
                parent,
                frozen_corpus,
                [],
            )
        )
    common.require(
        manifest["phase"] in ("A_ONLY", "B_ONLY"),
        "E_V22_PHASE: readiness requires a model phase",
    )
    result = v22.evaluate_root(
        root,
        v22.MANIFEST_NAME,
        contract,
        contract_raw,
        candidate,
        candidate_raw,
        parent,
        frozen_corpus,
        prior_results,
    )
    expected = (
        "MODEL_A_QUALIFICATION_PASS"
        if manifest["phase"] == "A_ONLY"
        else "MODEL_B_QUALIFICATION_PASS"
    )
    common.exact(result["status"], expected, "E_V22_STATUS")
    return result, manifest, manifest_raw


def artifact_map(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = snapshot["artifacts"]
    common.require(type(rows) is list and len(rows) == 5, "E_ARTIFACT_COUNT")
    result = {}
    for index, row in enumerate(rows):
        field = f"artifact_snapshot.artifacts[{index}]"
        row = common.exact_keys(
            row,
            {
                "bytes",
                "endpoint",
                "path",
                "sha256",
                "stat",
            },
            field,
        )
        endpoint = common.string(row["endpoint"], f"{field}.endpoint")
        path = common.string(row["path"], f"{field}.path")
        key = common.artifact_key(endpoint, path)
        common.require(key not in result, f"E_ARTIFACT_REUSE: {key}")
        common.integer(row["bytes"], f"{field}.bytes", 1)
        common.digest(row["sha256"], f"{field}.sha256")
        stat = common.stat_record(row["stat"], f"{field}.stat")
        common.exact(stat["size"], row["bytes"], f"{field}.stat.size")
        result[key] = row
    return result


def expected_artifacts(
    model: dict[str, Any],
    route_lock: dict[str, Any],
) -> dict[str, tuple[int, str]]:
    return {
        common.artifact_key("cuda", route_lock["cuda_model_path"]): (
            model["artifact"]["bytes"],
            model["artifact"]["sha256"],
        ),
        common.artifact_key("op15", route_lock["op15_shard_path"]): (
            route_lock["op15_shard_bytes"],
            route_lock["op15_shard_sha256"],
        ),
        common.artifact_key("op12", route_lock["op12_shard_path"]): (
            route_lock["op12_shard_bytes"],
            route_lock["op12_shard_sha256"],
        ),
    }


def validate_artifact_snapshot(
    snapshot: dict[str, Any],
    phase: str,
    model: dict[str, Any],
    route_lock: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    common.exact_keys(
        snapshot,
        {
            "artifacts",
            "completed_ns",
            "model_id",
            "phase",
            "route_lock_sha256",
            "schema",
            "slot",
            "started_ns",
        },
        "artifact_snapshot",
    )
    common.exact(
        snapshot["schema"],
        "s39-cp0-r1-artifact-snapshot-v2.3",
        "artifact_snapshot.schema",
    )
    common.exact(snapshot["phase"], phase, "artifact_snapshot.phase")
    common.exact(snapshot["slot"], model["slot"], "artifact_snapshot.slot")
    common.exact(
        snapshot["model_id"],
        model["model_id"],
        "artifact_snapshot.model",
    )
    common.digest(snapshot["route_lock_sha256"], "artifact_snapshot.route_lock")
    started = common.integer(snapshot["started_ns"], "artifact_snapshot.started", 1)
    completed = common.integer(
        snapshot["completed_ns"],
        "artifact_snapshot.completed",
        1,
    )
    common.require(started < completed, "E_ARTIFACT_INTERVAL")
    artifacts = artifact_map(snapshot)
    expected = expected_artifacts(model, route_lock)
    worker_endpoints = {"op12_worker", "op15_worker"}
    actual_worker_endpoints = {
        row["endpoint"]
        for row in artifacts.values()
        if row["endpoint"] in worker_endpoints
    }
    common.exact(
        actual_worker_endpoints,
        worker_endpoints,
        "artifact_snapshot.worker_endpoints",
    )
    common.exact(
        len(artifacts) - len(expected),
        2,
        "artifact_snapshot.worker_count",
    )
    common.require(
        set(expected).issubset(artifacts),
        "E_ARTIFACT_PATHS",
    )
    for key, (size, digest) in expected.items():
        common.exact(artifacts[key]["bytes"], size, f"E_ARTIFACT_BYTES: {key}")
        common.exact(
            artifacts[key]["sha256"],
            digest,
            f"E_ARTIFACT_SHA256: {key}",
        )
    return artifacts


def validate_readiness_lock(
    lock: dict[str, Any],
    manifest: dict[str, Any],
    phase_lock_sha256: str,
    artifact_raw: bytes,
    artifact_completed_ns: int,
) -> None:
    common.exact_keys(
        lock,
        {
            "artifact_snapshot_sha256",
            "event_ns",
            "phase",
            "phase_id",
            "schema",
            "v2_2_phase_lock_sha256",
        },
        "readiness_lock",
    )
    common.exact(
        lock["schema"],
        "s39-cp0-r1-readiness-lock-v2.3",
        "readiness_lock.schema",
    )
    common.exact(lock["phase"], manifest["phase"], "readiness_lock.phase")
    common.exact(lock["phase_id"], manifest["phase_id"], "readiness_lock.phase_id")
    common.exact(
        lock["artifact_snapshot_sha256"],
        common.sha256_bytes(artifact_raw),
        "readiness_lock.artifact_snapshot",
    )
    common.exact(
        lock["v2_2_phase_lock_sha256"],
        phase_lock_sha256,
        "readiness_lock.v2_2_phase_lock",
    )
    event_ns = common.integer(lock["event_ns"], "readiness_lock.event", 1)
    common.require(
        artifact_completed_ns <= event_ns < manifest["acquisition_started_ns"],
        "E_READINESS_LOCK_ORDER",
    )


def parse_artifact_stats(
    value: Any,
    field: str,
) -> dict[str, dict[str, Any]]:
    common.require(type(value) is list and len(value) == 5, f"E_TYPE: {field}")
    result = {}
    for index, item in enumerate(value):
        name = f"{field}[{index}]"
        item = common.exact_keys(item, {"endpoint", "path", "stat"}, name)
        key = common.artifact_key(
            common.string(item["endpoint"], f"{name}.endpoint"),
            common.string(item["path"], f"{name}.path"),
        )
        common.require(key not in result, f"E_FRESH_ARTIFACT_REUSE: {key}")
        common.stat_record(item["stat"], f"{name}.stat")
        result[key] = item
    return result


def validate_fresh_snapshot(
    fresh: dict[str, Any],
    fresh_raw: bytes,
    lock_raw: bytes,
    lock: dict[str, Any],
    manifest: dict[str, Any],
    contract: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    del fresh_raw
    common.exact_keys(
        fresh,
        {
            "artifact_stats",
            "completed_ns",
            "cuda",
            "phase",
            "phase_id",
            "phones",
            "readiness_lock_sha256",
            "schema",
            "started_ns",
        },
        "fresh_snapshot",
    )
    common.exact(
        fresh["schema"],
        "s39-cp0-r1-fresh-identity-v2.3",
        "fresh_snapshot.schema",
    )
    common.exact(fresh["phase"], manifest["phase"], "fresh_snapshot.phase")
    common.exact(fresh["phase_id"], manifest["phase_id"], "fresh_snapshot.phase_id")
    common.exact(
        fresh["readiness_lock_sha256"],
        common.sha256_bytes(lock_raw),
        "fresh_snapshot.readiness_lock",
    )
    started = common.integer(fresh["started_ns"], "fresh_snapshot.started", 1)
    completed = common.integer(fresh["completed_ns"], "fresh_snapshot.completed", 1)
    common.require(lock["event_ns"] <= started < completed, "E_FRESH_PRELOCK")
    common.require(
        completed < manifest["acquisition_started_ns"],
        "E_FRESH_ACQUISITION_ORDER",
    )
    common.require(
        manifest["acquisition_started_ns"] - completed
        <= contract["readiness_v2_3"]["fresh_snapshot_maximum_age_ns"],
        "E_FRESH_STALE",
    )
    stats = parse_artifact_stats(fresh["artifact_stats"], "fresh_snapshot.artifacts")
    common.exact(sorted(stats), sorted(artifacts), "fresh_snapshot.artifact_paths")
    for key in sorted(stats):
        common.exact(
            stats[key]["stat"],
            artifacts[key]["stat"],
            f"E_ARTIFACT_CHANGED: {key}",
        )

    cuda = common.exact_keys(
        fresh["cuda"],
        {
            "host",
            "host_boot_id",
            "memory_total_bytes",
            "name",
            "pci_bus_id",
            "uuid",
        },
        "fresh_snapshot.cuda",
    )
    expected_cuda = contract["readiness_v2_3"]["cuda_identity"]
    for key in ("host", "memory_total_bytes", "name", "uuid"):
        common.exact(cuda[key], expected_cuda[key], f"E_CUDA_IDENTITY: {key}")
    common.uuid(cuda["host_boot_id"], "fresh_snapshot.cuda.host_boot_id")
    common.string(cuda["pci_bus_id"], "fresh_snapshot.cuda.pci_bus_id")

    phones = fresh["phones"]
    common.exact_keys(phones, {"op12", "op15"}, "fresh_snapshot.phones")
    for phone in ("op15", "op12"):
        value = common.exact_keys(
            phones[phone],
            {
                "available_bytes",
                "boot_id",
                "device",
                "interfaces",
                "model",
                "product",
                "serial",
                "swap_total_bytes",
                "swap_used_bytes",
                "thermal_status",
            },
            f"fresh_snapshot.phones.{phone}",
        )
        expected = contract["readiness_v2_3"]["phone_identity"][phone]
        for key in ("device", "model", "product", "serial"):
            common.exact(value[key], expected[key], f"E_PHONE_IDENTITY: {phone}.{key}")
        common.uuid(value["boot_id"], f"fresh_snapshot.phones.{phone}.boot_id")
        common.require(
            common.integer(
                value["available_bytes"],
                f"fresh_snapshot.phones.{phone}.available",
            )
            >= contract["readiness_v2_3"]["phone_minimum_available_bytes"],
            f"E_PHONE_HEADROOM: {phone}",
        )
        common.integer(
            value["swap_total_bytes"],
            f"fresh_snapshot.phones.{phone}.swap_total",
        )
        common.exact(
            value["swap_used_bytes"],
            0,
            f"E_PHONE_SWAP: {phone}",
        )
        common.exact(value["thermal_status"], 0, f"E_PHONE_THERMAL_STATUS: {phone}")
        interfaces = value["interfaces"]
        common.require(type(interfaces) is dict and bool(interfaces), f"E_INTERFACE: {phone}")
        for interface, counters in interfaces.items():
            common.string(interface, f"interfaces.{phone}.name")
            common.exact_keys(
                counters,
                {"ipv4", "rx_bytes", "tx_bytes"},
                f"interfaces.{phone}.{interface}",
            )
            common.string(counters["ipv4"], f"interfaces.{phone}.{interface}.ipv4")
            common.integer(
                counters["rx_bytes"],
                f"interfaces.{phone}.{interface}.rx_bytes",
            )
            common.integer(
                counters["tx_bytes"],
                f"interfaces.{phone}.{interface}.tx_bytes",
            )
    return {"cuda": cuda, "phones": phones, "completed_ns": completed}


def manifest_digests(manifest: dict[str, Any]) -> dict[str, str]:
    result = {}
    for artifact in manifest["artifacts"]:
        result[artifact["role"]] = artifact["sha256"]
    return result


def validate_runtime_identity(
    runtime: dict[str, Any],
    runtime_raw: bytes,
    fresh_raw: bytes,
    fresh_derived: dict[str, Any],
    manifest: dict[str, Any],
    contract: dict[str, Any],
    artifact_digests: dict[str, str],
    direct_payload_bytes: int,
    model_id: str,
) -> dict[str, Any]:
    del runtime_raw
    common.exact_keys(
        runtime,
        {
            "completed_ns",
            "executors",
            "fresh_snapshot_sha256",
            "phase",
            "phase_id",
            "route_epoch",
            "schema",
            "started_ns",
        },
        "runtime_identity",
    )
    common.exact(
        runtime["schema"],
        "s39-cp0-r1-runtime-identity-v2.3",
        "runtime_identity.schema",
    )
    common.exact(runtime["phase"], manifest["phase"], "runtime_identity.phase")
    common.exact(
        runtime["phase_id"],
        manifest["phase_id"],
        "runtime_identity.phase_id",
    )
    common.exact(
        runtime["fresh_snapshot_sha256"],
        common.sha256_bytes(fresh_raw),
        "runtime_identity.fresh_snapshot",
    )
    route_epoch = common.integer(runtime["route_epoch"], "runtime_identity.epoch", 1)
    started = common.integer(runtime["started_ns"], "runtime_identity.started", 1)
    completed = common.integer(runtime["completed_ns"], "runtime_identity.completed", 1)
    common.require(
        manifest["acquisition_started_ns"] <= started < completed
        <= manifest["phase_closed_ns"],
        "E_RUNTIME_INTERVAL",
    )
    executors = runtime["executors"]
    common.require(type(executors) is list and len(executors) == 3, "E_EXECUTOR_COUNT")
    by_id = {}
    for index, executor in enumerate(executors):
        field = f"runtime_identity.executors[{index}]"
        executor_id = common.string(executor.get("executor_id"), f"{field}.id")
        common.require(executor_id not in by_id, f"E_EXECUTOR_REUSE: {executor_id}")
        by_id[executor_id] = executor
    common.exact(sorted(by_id), ["GPU", "PHONE_OP12", "PHONE_OP15"], "executors")

    gpu = common.exact_keys(
        by_id["GPU"],
        {
            "artifact_path",
            "artifact_sha256",
            "artifact_stat",
            "executor_id",
            "gpu_uuid",
            "host_boot_id",
            "model_id",
            "route_epoch",
        },
        "executor.GPU",
    )
    common.exact(gpu["gpu_uuid"], fresh_derived["cuda"]["uuid"], "E_RUNTIME_GPU_UUID")
    common.exact(
        gpu["host_boot_id"],
        fresh_derived["cuda"]["host_boot_id"],
        "E_RUNTIME_HOST_BOOT",
    )
    common.exact(gpu["model_id"], model_id, "E_RUNTIME_GPU_MODEL")
    common.exact(gpu["route_epoch"], route_epoch, "E_RUNTIME_GPU_EPOCH")
    gpu_key = common.artifact_key("cuda", gpu["artifact_path"])
    common.require(gpu_key in artifact_digests["loaded_artifacts"], "E_RUNTIME_GPU_PATH")
    expected_gpu_artifact = artifact_digests["loaded_artifacts"][gpu_key]
    common.exact(
        gpu["artifact_sha256"],
        expected_gpu_artifact["sha256"],
        "E_RUNTIME_GPU_ARTIFACT",
    )
    common.stat_record(gpu["artifact_stat"], "executor.GPU.artifact_stat")
    common.exact(
        gpu["artifact_stat"],
        expected_gpu_artifact["stat"],
        "E_RUNTIME_GPU_STAT",
    )

    mechanics_role = f"model.{model_id}.mechanics.phone"
    for phone in ("op15", "op12"):
        executor_id = f"PHONE_{phone.upper()}"
        value = common.exact_keys(
            by_id[executor_id],
            {
                "active_sequences_after_cleanup",
                "available_bytes",
                "boot_id",
                "direct_peer",
                "executor_id",
                "gpu_max_millic",
                "interface_after",
                "interface_before",
                "loaded_shard_path",
                "loaded_shard_sha256",
                "loaded_shard_stat",
                "mechanics_sha256",
                "model_id",
                "placement_sha256",
                "process_swap_bytes",
                "route_epoch",
                "route_transfer_sha256",
                "serial",
                "worker_model_sha256",
                "worker_boot_nonce",
                "worker_executable_path",
                "worker_executable_sha256",
                "worker_executable_stat",
                "worker_pid",
                "worker_start_ticks",
                "session_protocol_version",
            },
            f"executor.{executor_id}",
        )
        fresh_phone = fresh_derived["phones"][phone]
        common.exact(value["serial"], fresh_phone["serial"], f"E_RUNTIME_SERIAL: {phone}")
        common.exact(value["boot_id"], fresh_phone["boot_id"], f"E_RUNTIME_BOOT: {phone}")
        common.exact(value["model_id"], model_id, f"E_RUNTIME_MODEL: {phone}")
        common.exact(value["route_epoch"], route_epoch, f"E_RUNTIME_EPOCH: {phone}")
        common.integer(value["worker_pid"], f"executor.{phone}.worker_pid", 1)
        common.integer(
            value["worker_start_ticks"],
            f"executor.{phone}.worker_start_ticks",
            1,
        )
        nonce = common.string(
            value["worker_boot_nonce"],
            f"executor.{phone}.worker_boot_nonce",
        )
        common.require(
            len(nonce) == 16
            and all(character in "0123456789abcdef" for character in nonce),
            f"E_RUNTIME_WORKER_NONCE: {phone}",
        )
        common.exact(
            value["session_protocol_version"],
            2,
            f"E_RUNTIME_SESSION_PROTOCOL: {phone}",
        )
        common.exact(
            value["worker_model_sha256"],
            expected_gpu_artifact["sha256"],
            f"E_RUNTIME_WORKER_MODEL: {phone}",
        )
        worker_key = common.artifact_key(
            f"{phone}_worker",
            common.string(
                value["worker_executable_path"],
                f"executor.{phone}.worker_executable_path",
            ),
        )
        common.require(
            worker_key in artifact_digests["loaded_artifacts"],
            f"E_RUNTIME_WORKER_PATH: {phone}",
        )
        expected_worker = artifact_digests["loaded_artifacts"][worker_key]
        common.exact(
            value["worker_executable_sha256"],
            expected_worker["sha256"],
            f"E_RUNTIME_WORKER_DIGEST: {phone}",
        )
        common.stat_record(
            value["worker_executable_stat"],
            f"executor.{phone}.worker_executable_stat",
        )
        common.exact(
            value["worker_executable_stat"],
            expected_worker["stat"],
            f"E_RUNTIME_WORKER_STAT: {phone}",
        )
        shard_key = common.artifact_key(phone, value["loaded_shard_path"])
        common.require(
            shard_key in artifact_digests["loaded_artifacts"],
            f"E_RUNTIME_SHARD_PATH: {phone}",
        )
        expected_shard = artifact_digests["loaded_artifacts"][shard_key]
        common.exact(
            value["loaded_shard_sha256"],
            expected_shard["sha256"],
            f"E_RUNTIME_SHARD_DIGEST: {phone}",
        )
        common.stat_record(
            value["loaded_shard_stat"],
            f"executor.{phone}.loaded_shard_stat",
        )
        common.exact(
            value["loaded_shard_stat"],
            expected_shard["stat"],
            f"E_RUNTIME_SHARD_STAT: {phone}",
        )
        common.exact(
            value["mechanics_sha256"],
            artifact_digests[mechanics_role],
            f"E_RUNTIME_MECHANICS_LINK: {phone}",
        )
        placement_role = f"model.{model_id}.placement.{phone}"
        common.exact(
            value["placement_sha256"],
            artifact_digests[placement_role],
            f"E_RUNTIME_PLACEMENT_LINK: {phone}",
        )
        common.exact(
            value["route_transfer_sha256"],
            artifact_digests[f"model.{model_id}.route_transfer"],
            f"E_RUNTIME_TRANSFER_LINK: {phone}",
        )
        common.require(
            common.integer(value["available_bytes"], f"executor.{phone}.available")
            >= contract["readiness_v2_3"]["phone_minimum_available_bytes"],
            f"E_RUNTIME_HEADROOM: {phone}",
        )
        common.exact(
            value["process_swap_bytes"],
            contract["readiness_v2_3"]["maximum_process_swap_bytes"],
            f"E_RUNTIME_PROCESS_SWAP: {phone}",
        )
        common.exact(
            value["active_sequences_after_cleanup"],
            0,
            f"E_RUNTIME_CLEANUP: {phone}",
        )
        common.require(
            common.integer(
                value["gpu_max_millic"],
                f"executor.{phone}.gpu_max_millic",
                1,
            )
            <= contract["readiness_v2_3"]["phone_maximum_gpu_temp_millic"],
            f"E_RUNTIME_THERMAL: {phone}",
        )
        before = common.exact_keys(
            value["interface_before"],
            {"interface", "rx_bytes", "tx_bytes"},
            f"executor.{phone}.interface_before",
        )
        after = common.exact_keys(
            value["interface_after"],
            {"interface", "rx_bytes", "tx_bytes"},
            f"executor.{phone}.interface_after",
        )
        common.exact(after["interface"], before["interface"], f"E_INTERFACE_NAME: {phone}")
        common.require(
            before["interface"] in fresh_phone["interfaces"],
            f"E_FRESH_INTERFACE: {phone}",
        )
        fresh_interface = fresh_phone["interfaces"][before["interface"]]
        for counter in ("rx_bytes", "tx_bytes"):
            common.integer(before[counter], f"executor.{phone}.before.{counter}")
            common.integer(after[counter], f"executor.{phone}.after.{counter}")
            common.exact(
                before[counter],
                fresh_interface[counter],
                f"E_FRESH_INTERFACE_COUNTER: {phone}.{counter}",
            )
            common.require(
                after[counter] >= before[counter],
                f"E_INTERFACE_COUNTER_RESET: {phone}.{counter}",
            )
        peer = common.exact_keys(
            value["direct_peer"],
            {
                "interface",
                "local_ipv4",
                "peer_ipv4",
                "socket_peer_observed",
            },
            f"executor.{phone}.peer",
        )
        common.exact(peer["interface"], before["interface"], f"E_PEER_INTERFACE: {phone}")
        common.exact(peer["socket_peer_observed"], True, f"E_PEER_SOCKET: {phone}")
        common.exact(
            peer["local_ipv4"],
            fresh_interface["ipv4"],
            f"E_FRESH_LOCAL_IP: {phone}",
        )
        common.string(peer["peer_ipv4"], f"executor.{phone}.peer.remote")
        if phone == "op15":
            common.require(
                after["tx_bytes"] - before["tx_bytes"] >= direct_payload_bytes,
                "E_INTERFACE_TRANSFER_BYTES: op15.tx",
            )
        else:
            common.require(
                after["rx_bytes"] - before["rx_bytes"] >= direct_payload_bytes,
                "E_INTERFACE_TRANSFER_BYTES: op12.rx",
            )
    common.exact(
        by_id["PHONE_OP15"]["direct_peer"]["peer_ipv4"],
        by_id["PHONE_OP12"]["direct_peer"]["local_ipv4"],
        "E_DIRECT_PEER: op15->op12",
    )
    common.exact(
        by_id["PHONE_OP12"]["direct_peer"]["peer_ipv4"],
        by_id["PHONE_OP15"]["direct_peer"]["local_ipv4"],
        "E_DIRECT_PEER: op12->op15",
    )
    return {"executor_count": 3, "route_epoch": route_epoch}


def validate_readiness(
    contract: dict[str, Any],
    candidate: dict[str, Any],
    bundle_root: Path,
    artifact_path: Path,
    readiness_lock_path: Path,
    fresh_path: Path,
    runtime_path: Path,
    *,
    candidate_path: Path = DEFAULT_CANDIDATE,
    a_bundle_root: Path | None = None,
) -> dict[str, Any]:
    v22_result, manifest, manifest_raw = reevaluate_v22_bundle(
        bundle_root,
        candidate_path,
        a_bundle_root,
    )
    slot = "A" if manifest["phase"] == "A_ONLY" else "B"
    model = next(model for model in candidate["models"] if model["slot"] == slot)
    digests = manifest_digests(manifest)
    route_role = f"model.{model['model_id']}.route_lock"
    route_raw = v2.secure_read(
        bundle_root,
        next(
            artifact["path"]
            for artifact in manifest["artifacts"]
            if artifact["role"] == route_role
        ),
    )
    route_rows = v2.parse_jsonl(route_raw, route_role, manifest["phase_id"])
    common.require(len(route_rows) == 1, "E_ROUTE_LOCK_ROWS")
    route_lock = route_rows[0]

    artifact, artifact_raw = common.read_canonical(artifact_path)
    artifacts = validate_artifact_snapshot(
        artifact,
        manifest["phase"],
        model,
        route_lock,
    )
    common.exact(
        artifact["route_lock_sha256"],
        digests[route_role],
        "artifact_snapshot.route_lock",
    )
    phase_lock_sha256 = digests["phase.lock"]
    lock, lock_raw = common.read_canonical(readiness_lock_path)
    validate_readiness_lock(
        lock,
        manifest,
        phase_lock_sha256,
        artifact_raw,
        artifact["completed_ns"],
    )
    fresh, fresh_raw = common.read_canonical(fresh_path)
    fresh_derived = validate_fresh_snapshot(
        fresh,
        fresh_raw,
        lock_raw,
        lock,
        manifest,
        contract,
        artifacts,
    )
    runtime, runtime_raw = common.read_canonical(runtime_path)
    runtime_derived = validate_runtime_identity(
        runtime,
        runtime_raw,
        fresh_raw,
        fresh_derived,
        manifest,
        contract,
        {**digests, "loaded_artifacts": artifacts},
        v22_result["derived"]["model"]["transfer"]["direct_payload_bytes"],
        model["model_id"],
    )
    return {
        "artifact_snapshot_sha256": common.sha256_bytes(artifact_raw),
        "fresh_snapshot_sha256": common.sha256_bytes(fresh_raw),
        "model_id": model["model_id"],
        "phase": manifest["phase"],
        "phase_id": manifest["phase_id"],
        "readiness_lock_sha256": common.sha256_bytes(lock_raw),
        "runtime": runtime_derived,
        "runtime_identity_sha256": common.sha256_bytes(runtime_raw),
        "v2_2_bundle_manifest_sha256": common.sha256_bytes(manifest_raw),
        "v2_2_result_sha256": common.sha256_bytes(
            v2.canonical_bytes(v22_result)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CP0-R1 V2.3 readiness")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--artifact-snapshot", type=Path, required=True)
    parser.add_argument("--readiness-lock", type=Path, required=True)
    parser.add_argument("--fresh-snapshot", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--a-bundle-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract, _, candidate, _ = validate_inputs(args.contract, args.candidate)
        derived = validate_readiness(
            contract,
            candidate,
            args.bundle_root,
            args.artifact_snapshot,
            args.readiness_lock,
            args.fresh_snapshot,
            args.runtime_identity,
            candidate_path=args.candidate,
            a_bundle_root=args.a_bundle_root,
        )
        print(common.canonical_bytes({
            "derived": derived,
            "schema": "s39-cp0-r1-readiness-result-v2.3",
            "status": "V2_3_READINESS_PASS",
        }).decode("ascii"), end="")
        return 0
    except (
        common.ReadinessError,
        v2.EvidenceError,
        OSError,
        KeyError,
        ValueError,
    ) as exc:
        print(f"CP0_R1_V2_3_REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
