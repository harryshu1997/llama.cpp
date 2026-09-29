#!/usr/bin/env python3
"""Validate the unified RTX 4060 Ti and OP15 campaign without inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler.campaigns.burstgpt.runner import (  # noqa: E402
    GEMMA_ROLE,
    QWEN_ROLE,
    _ffn_shard_storage,
    apply_named_replay_schedule,
    artifact_aliases,
    digest,
    load_object,
    merge_rows,
    trace_role,
    validate_trace,
)
from research_dev.scheduler import (  # noqa: E402
    BackgroundRuntimeMonitor,
    DeviceMemoryCapacity,
    ModelManifest,
    Request,
    RuntimeCapabilityCatalog,
    RuntimePlacementSnapshot,
    RuntimeProtectedWorkObservation,
    UnifiedScheduler,
)
from research_dev.scheduler.adapters import (  # noqa: E402
    AndroidUsbRestorationReceipt,
    catalog_preloaded_residency_samples,
    DeviceRuntimeTelemetry,
    EndpointRuntimeSample,
    ExecutorResidencySample,
    LinuxHostRuntimeProbe,
    materialize_preflight_executor_samples,
    nvidia_gpu_snapshot,
    phone_sessions_from_json,
    PhysicalAdapterError,
    PhysicalPreflightCheck,
    PhysicalPreflightModel,
    UnifiedRuntimeSnapshotBuilder,
    probe_functionfs_usb_device,
    probe_llama_endpoint,
    probe_phone_power_with_adb_fallback,
    probe_phone_runtime_with_adb_fallback,
    rapl_package_snapshot,
    run_physical_preflight,
)
from research_dev.scheduler.adapters.phone_helpers import UsbTopologyCheck, check_usb_topology  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.arguments import (  # noqa: E402
    device_power_json,
    elastic_phones_json,
    speculative_rows_json,
)
from research_dev.scheduler.adapters.speculative_rows import (  # noqa: E402
    DRAFT_BYTES_PARAMETER,
    DRAFT_SHA256_PARAMETER,
    speculative_adapter_parameters,
)
from research_dev.scheduler.configuration.campaign import speculative_rows_configuration  # noqa: E402
from research_dev.scheduler._internal.model_manifest import _sha256_file  # noqa: E402
from research_dev.scheduler.adapters.device_power import device_power_capability  # noqa: E402
from research_dev.scheduler.configuration.campaign import DevicePowerConfiguration  # noqa: E402
from research_dev.scheduler.adapters.ffn_shards import (  # noqa: E402
    FfnShardIndex,
    FfnShardIndexError,
    remote_hash_entries as ffn_shard_hash_entries,
    verify_remote_hashes as verify_ffn_shard_hashes,
)


EXPECTED_GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4060 Ti"
EXPECTED_GPU_CAPACITY_BYTES = 17_175_674_880
PREFLIGHT_RESULT_SCHEMA = "s42-unified-fp16-physical-preflight-v1"


class RigPreflightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RigPreflightError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def check(
    check_id: str,
    condition: bool,
    detail: str,
    *,
    advisory: bool = False,
) -> PhysicalPreflightCheck:
    return PhysicalPreflightCheck(
        check_id,
        "PASS" if condition else "WARN" if advisory else "BLOCKED",
        detail,
    )


def _retry_raw_observation(
    probe,
    accepted,
    *,
    timeout_s: float = 10,
):
    if not callable(probe) or not callable(accepted) or timeout_s <= 0:
        raise RigPreflightError("raw observation retry is invalid")
    deadline = time.monotonic() + timeout_s
    while True:
        observed = probe()
        if accepted(observed) or time.monotonic() >= deadline:
            return observed
        time.sleep(0.25)


def _refresh_background_observations(
    monitor: BackgroundRuntimeMonitor,
    names: tuple[str, ...],
    *,
    timeout_s: float = 15,
) -> dict[str, object | None]:
    if not names or timeout_s <= 0:
        raise RigPreflightError("background observation refresh is invalid")
    requested_at_ns = time.monotonic_ns()
    monitor.request_refresh()
    deadline = time.monotonic() + timeout_s
    while True:
        snapshots = {name: monitor.snapshot(name) for name in names}
        if all(
            row.captured_at_ns >= requested_at_ns
            for row in snapshots.values()
        ):
            return {
                name: (
                    None
                    if row.error is not None or row.stale
                    else row.value
                )
                for name, row in snapshots.items()
            }
        if time.monotonic() >= deadline:
            return {name: None for name in names}
        time.sleep(0.05)


def _local_port_available(
    endpoint: str,
    sample: EndpointRuntimeSample | None = None,
) -> tuple[bool, str]:
    parsed = urlsplit(endpoint)
    live = probe_llama_endpoint(endpoint) if sample is None else sample
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return live.ready, (
            "remote endpoint health="
            + live.health
            + " slots_probe="
            + live.slots_probe
            + " free_slots="
            + str(live.free_slots)
        )
    if parsed.port is None:
        return False, "endpoint port is absent"
    if live.ready:
        return True, "endpoint is live"
    family = socket.AF_INET6 if ":" in parsed.hostname else socket.AF_INET
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        probe.bind((parsed.hostname, parsed.port))
        return True, "endpoint port is free for canonical launch"
    except OSError as error:
        return False, "endpoint port conflict: " + str(error)
    finally:
        probe.close()


def _profile_features(
    catalog: RuntimeCapabilityCatalog,
    manifest: ModelManifest,
    request: Request,
) -> dict[str, int]:
    profiles = [
        row
        for row in catalog.route_shape_profiles
        if row.artifact_sha256 == manifest.artifact_sha256
        and row.minimum_input_tokens <= request.input_tokens
            <= row.maximum_input_tokens
        and row.minimum_output_tokens <= request.output_tokens
            <= row.maximum_output_tokens
        and all(
            catalog.placement_profile.devices[device_id].kind
                in {"cpu", "gpu"}
            for device_id in row.device_ids
        )
    ]
    result = {
        "active_cpu_requests": 0,
        "active_cpu_slots": 0,
        "actual_batch_size": min(request.input_tokens, 512),
        "large_model_op15": 0,
        "large_phase_id": 0,
        "memory_bandwidth_pressure_basis_points": 0,
        "memory_stall_avg10_basis_points": 0,
        "prompt_ubatch_count": (request.input_tokens + 511) // 512,
    }
    if profiles:
        profile = min(profiles, key=lambda row: row.specificity)
        result.update({
            name: bounds[0]
            for name, bounds in profile.feature_ranges.items()
        })
    return result


def _trace_request_for_row(row: dict[str, Any], model_id: str) -> Request:
    """The exact request the runner will submit for this trace row."""
    return Request(
        request_id=str(row["event_id"]),
        workload_id="physical:" + model_id,
        arrival_us=int(row["arrival_us"]),
        deadline_us=int(row["arrival_us"]) + int(row["slo_us"]),
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        quality_requirement="semantic",
    )


def _request_for_row(row: dict[str, Any], model_id: str) -> Request:
    return Request(
        request_id="preflight:" + model_id,
        workload_id="physical:" + model_id,
        arrival_us=1,
        deadline_us=1 + int(row["slo_us"]),
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        quality_requirement="semantic",
    )


def _precision_check(
    check_id: str,
    manifest: ModelManifest,
    matrix_quantization: str,
    *,
    allowed_unquantized_matrix_ids: tuple[str, ...] = (),
) -> PhysicalPreflightCheck:
    matrix = tuple(row for row in manifest.tensors if len(row.shape) >= 2)
    allowed = set(allowed_unquantized_matrix_ids)
    matrix_ids = {row.tensor_id for row in matrix}
    invalid_matrix = tuple(
        row.tensor_id
        for row in matrix
        if (
            row.quantization != matrix_quantization
            and row.tensor_id not in allowed
        )
    )
    invalid_f32 = tuple(
        row.tensor_id
        for row in manifest.tensors
        if (
            row.quantization == "F32"
            and len(row.shape) >= 2
            and row.tensor_id not in allowed
        )
    )
    missing_allowed = tuple(sorted(allowed - matrix_ids))
    return check(
        check_id,
        bool(matrix)
        and not invalid_matrix
        and not invalid_f32
        and not missing_allowed,
        (
            str(len(matrix))
            + " matrix tensors require "
            + matrix_quantization
            + "; invalid="
            + ",".join((*invalid_matrix, *invalid_f32)[:8])
            + "; allowed_unquantized="
            + ",".join(sorted(allowed) or ("none",))
            + "; missing_allowed="
            + ",".join(missing_allowed or ("none",))
        ),
    )


def _phone_session_checks(
    catalog: RuntimeCapabilityCatalog,
    artifact_sha256s: set[str],
) -> tuple[PhysicalPreflightCheck, ...]:
    by_id = {}
    inconsistent = set()
    for executor in catalog.executors:
        for session in executor.phone_sessions:
            current = by_id.get(session.session_id)
            if current is not None and current != session:
                inconsistent.add(session.session_id)
            by_id[session.session_id] = session
    sessions = tuple(by_id.values())
    required = {"HTP0", "HTP1", "HTP2"}
    shared_compute = {
        session.shared_compute_resource_id for session in sessions
    }
    shared_transport = {
        session.shared_transport_resource_ids for session in sessions
    }
    memory_resources = {
        session.memory_resource_id for session in sessions
    }
    resource_ids = set(catalog.resources)
    shared_ids = set(shared_compute)
    for row in shared_transport:
        shared_ids.update(row)
    resource_profiles = tuple(
        catalog.resources[resource_id]
        for resource_id in sorted(shared_ids & resource_ids)
    )
    resident_identity_ok = all(
        session.residency_state == "cold"
        or (
            session.resident_artifact_sha256 in artifact_sha256s
            and type(session.resident_geometry_sha256) is str
        )
        for session in sessions
    )
    return (
        check(
            "phone-session-discovery",
            required <= set(by_id)
            and not inconsistent
            and all(by_id[session_id].ready for session_id in required),
            "ready sessions=" + ",".join(sorted(
                session_id for session_id, session in by_id.items()
                if session.ready
            )),
        ),
        check(
            "phone-session-residency-capacity",
            len(memory_resources) == len(sessions) >= 3
            and all(session.resident_memory_limit_bytes > 0 for session in sessions),
            "distinct session memory resources=" + str(len(memory_resources)),
        ),
        check(
            "phone-session-shared-resources",
            len(shared_compute) == 1
            and len(shared_transport) == 1
            and shared_ids <= resource_ids
            and all(profile.capacity == 1 for profile in resource_profiles),
            "shared compute=" + ",".join(sorted(shared_compute))
            + " shared transport="
            + ",".join(sorted(shared_ids - shared_compute)),
        ),
        check(
            "phone-session-residency-identity",
            resident_identity_ok,
            "resident session artifacts and shard geometry match the catalog",
        ),
    )


def _desktop_control_checks(
    catalog: RuntimeCapabilityCatalog,
    large_artifacts: set[str],
) -> tuple[PhysicalPreflightCheck, ...]:
    devices = catalog.placement_profile.devices
    controls = {
        row.artifact_sha256: row
        for row in catalog.desktop_control_profiles
    }
    rows = []
    for artifact_sha256 in sorted(large_artifacts):
        control = controls.get(artifact_sha256)
        gpu = False if control is None else any(
            devices[row.primary_device_id].kind == "gpu"
            for row in control.operator_placements
        )
        rows.append(check(
            "desktop-control-cuda:" + artifact_sha256[7:19],
            control is not None
            and control.maturity == "QUALIFIED"
            and gpu,
            "large-model desktop control is qualified and uses CUDA",
        ))
    return tuple(rows)


def validate_phone_router_deployment(
    binary: Path,
    receipt: Path,
) -> str:
    binary_sha256 = digest(binary)
    rows = []
    for line in receipt.read_text(encoding="ascii").splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, "phone router receipt row")
        rows.append(fields)
    require(
        len(rows) == 2
        and all(len(row[0]) == 64 for row in rows)
        and rows[0][0] == rows[1][0] == binary_sha256,
        "phone router deployment identity",
    )
    require(
        b"terminate_requested" in binary.read_bytes(),
        "phone router terminal-control capability",
    )
    return binary_sha256


def _require_phone_session_discovery(
    args: argparse.Namespace, catalog: RuntimeCapabilityCatalog
) -> None:
    catalog_sessions = {
        session.session_id: session
        for executor in catalog.executors
        for session in executor.phone_sessions
    }
    discovered_sessions = (
        tuple(catalog_sessions.values())
        if args.phone_session_discovery is None
        else phone_sessions_from_json(
            load_object(args.phone_session_discovery)
        )
    )
    require(
        len(discovered_sessions) >= 3
        and all(session.ready for session in discovered_sessions)
        and {
            session.session_id for session in discovered_sessions
        } == set(catalog_sessions)
        and all(
            catalog_sessions[session.session_id] == session
            for session in discovered_sessions
        ),
        "phone session discovery differs from the capability catalog",
    )


def _phone_endpoints(catalog: RuntimeCapabilityCatalog) -> tuple[str, ...]:
    return tuple(sorted({
        capability.endpoint
        for capability in catalog.executors
        if catalog.placement_profile.devices[
            capability.device_id
        ].kind == "phone"
        and urlsplit(capability.endpoint).scheme in {"http", "https"}
    }))


def _background_probes(
    args: argparse.Namespace, phone_endpoints: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "phone-runtime": lambda: probe_phone_runtime_with_adb_fallback(
            args.phone_diagnostic_endpoint,
            args.phone_usb_serial,
            args.adb_port,
        ),
        "phone-power": lambda: probe_phone_power_with_adb_fallback(
            args.phone_diagnostic_endpoint,
            args.phone_usb_serial,
            args.adb_port,
        ),
        **{
            "endpoint:" + endpoint: (
                lambda endpoint=endpoint: probe_llama_endpoint(endpoint)
            )
            for endpoint in phone_endpoints
        },
    }


def _register_models_under_monitor(
    args: argparse.Namespace,
    scheduler: UnifiedScheduler,
    model_paths: dict[str, Path],
    background_probes: dict[str, Any],
) -> tuple[dict[str, ModelManifest], dict[str, Any]]:
    background_monitor = BackgroundRuntimeMonitor(
        background_probes,
        refresh_interval_s=0.5,
        stale_after_s=5,
    )
    background_monitor.start()
    try:
        manifests = {
            model_id: scheduler.register_gguf_model(model_id, path)
            for model_id, path in model_paths.items()
        }
        scheduler.load_automated_observations(
            load_object(args.observation_store_input)
        )
        scheduler.load_adaptive_decode_observations(
            load_object(args.adaptive_observation_store_input)
        )
        background_values = _refresh_background_observations(
            background_monitor,
            tuple(background_probes),
        )
    finally:
        background_monitor.stop()
    return manifests, background_values


def _require_manifest_identity(
    manifests: dict[str, ModelManifest],
    expected_qwen: ModelManifest,
    expected_gemma: ModelManifest,
) -> None:
    for expected in (expected_qwen, expected_gemma):
        actual = manifests[expected.model_id]
        require(
            actual.artifact_sha256 == expected.artifact_sha256
            and actual.artifact_bytes == expected.artifact_bytes
            and actual.block_count == expected.block_count
            and actual.embedding_length == expected.embedding_length
            and actual.feed_forward_length == expected.feed_forward_length,
            "GGUF manifest identity: " + expected.model_id,
        )


def _phone_router_identity(args: argparse.Namespace) -> tuple[str, str]:
    if args.phone_router is None:
        require(
            args.phone_session_discovery is not None,
            "direct phone preflight requires session discovery",
        )
        phone_router_sha256 = "sha256:" + digest(args.phone_session_discovery)
        phone_router_detail = (
            "direct phone-session discovery SHA-256 is "
            + phone_router_sha256
        )
    else:
        assert args.phone_router_receipt is not None
        phone_router_sha256 = validate_phone_router_deployment(
            args.phone_router, args.phone_router_receipt
        )
        phone_router_detail = (
            "terminal-aware phone router SHA-256 is "
            + phone_router_sha256
        )
    return phone_router_sha256, phone_router_detail


def _model_checks(
    catalog: RuntimeCapabilityCatalog,
    manifests: dict[str, ModelManifest],
    expected_qwen: ModelManifest,
    expected_gemma: ModelManifest,
    llama_model_id: str,
) -> list[PhysicalPreflightCheck]:
    rig_checks = []
    rig_checks.extend((
        _precision_check("qwen-matrix-precision", manifests[
            expected_qwen.model_id
        ], "F16"),
        _precision_check("gemma-matrix-precision", manifests[
            expected_gemma.model_id
        ], "F16"),
        _precision_check(
            "llama-matrix-precision",
            manifests[llama_model_id],
            "Q4_0",
            allowed_unquantized_matrix_ids=("token_embd.weight",),
        ),
    ))
    rig_checks.extend(_desktop_control_checks(catalog, {
        expected_qwen.artifact_sha256,
        expected_gemma.artifact_sha256,
    }))
    rig_checks.extend(_phone_session_checks(
        catalog,
        {manifest.artifact_sha256 for manifest in manifests.values()},
    ))
    return rig_checks


def _ffn_shard_preflight(
    args: argparse.Namespace,
    expected_qwen: ModelManifest,
    expected_gemma: ModelManifest,
    expected_overlay: ModelManifest | None = None,
) -> tuple[dict[str, FfnShardIndex], tuple[PhysicalPreflightCheck, ...]]:
    indexes: dict[str, FfnShardIndex] = {}
    specs = (
        (expected_qwen.artifact_sha256, args.qwen_ffn_shards),
        (expected_gemma.artifact_sha256, args.gemma_ffn_shards),
    )
    if expected_overlay is not None:
        specs += ((expected_overlay.artifact_sha256, args.llama_ffn_shards),)
    for artifact, spec in specs:
        if spec is None:
            continue
        local, separator, remote_dir = spec.rpartition("=")
        require(
            bool(separator) and bool(local) and bool(remote_dir),
            "FFN shard spec must be LOCAL=PHONE_DIR",
        )
        try:
            index = FfnShardIndex.load(Path(local), remote_dir)
        except FfnShardIndexError as error:
            raise RigPreflightError(str(error)) from error
        require(
            index.parent_sha256 == artifact,
            "FFN shard index parent differs from model artifact",
        )
        indexes[artifact] = index
    if not indexes:
        return {}, ()
    entries = ffn_shard_hash_entries(indexes)
    paths = tuple(entries.values())
    command = "sha256sum " + " ".join(
        shlex.quote(path) for path in paths
    )
    try:
        completed = subprocess.run(
            (
                str(args.adb), "-P", str(args.adb_port),
                "-s", args.phone_usb_serial, "shell",
                "su", "-c", command,
            ),
            check=True,
            capture_output=True,
            text=True,
            encoding="ascii",
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RigPreflightError(
            "phone FFN shard hash verification failed: " + str(error)
        ) from error
    observed_by_path = {}
    for line in completed.stdout.splitlines():
        digest_text, separator, path = line.strip().partition("  ")
        if separator and len(digest_text) == 64:
            observed_by_path[path] = "sha256:" + digest_text
    remote_hashes = {
        key: observed_by_path.get(path, "")
        for key, path in entries.items()
    }
    try:
        verify_ffn_shard_hashes(indexes, remote_hashes)
    except FfnShardIndexError as error:
        raise RigPreflightError(str(error)) from error
    checks = tuple(
        check(
            "phone-ffn-shards:" + artifact[7:19],
            True,
            "index " + index.index_sha256 + " verified "
            + str(len(index.records)) + " deployed shards",
        )
        for artifact, index in sorted(indexes.items())
    )
    return indexes, checks


def _gpu_checks(gpu: dict[str, Any]) -> tuple[PhysicalPreflightCheck, ...]:
    gpu_identity_ok = (
        gpu["uuid"] == EXPECTED_GPU_UUID
        and gpu["name"] == EXPECTED_GPU_NAME
        and gpu["memory_total_bytes"] == EXPECTED_GPU_CAPACITY_BYTES
    )
    return (
        check(
            "gpu-identity",
            gpu_identity_ok,
            "GPU is " + str(gpu["name"]) + " " + str(gpu["uuid"]),
        ),
        check(
            "gpu-memory",
            0 < gpu["memory_free_bytes"] <= gpu["memory_total_bytes"],
            "GPU free bytes are " + str(gpu["memory_free_bytes"]),
        ),
    )


def _rapl_check() -> PhysicalPreflightCheck:
    try:
        rapl = rapl_package_snapshot()
    except (OSError, RuntimeError, ValueError) as error:
        rapl = None
        rapl_detail = "RAPL probe failed: " + str(error)
    else:
        rapl_detail = "RAPL package energy counter is readable"
    return check("cpu-energy-boundary", rapl is not None, rapl_detail)


def _device_power_checks(args: argparse.Namespace) -> list[PhysicalPreflightCheck]:
    """Advisory ``device-power-control:<device>``: the sudoers rules the device power controller
    needs (``sudo -n -l`` on its exact argv) and a readable EPP. WARN only: the runner degrades
    to UNAVAILABLE (policy off) on its own; absent policy adds no check."""
    policy = getattr(args, "device_power", None)
    if policy is None:
        return []
    configuration = DevicePowerConfiguration.from_json(dict(policy))
    available, detail, _rows = device_power_capability(configuration)
    return [check("device-power-control:" + configuration.device, available, detail, advisory=True)]


def _speculative_rows_checks(args: argparse.Namespace, manifests: dict[str, Any]) -> list[PhysicalPreflightCheck]:
    """``speculative-rows:<model>``: the draft GGUF exists and shares the target vocabulary and,
    when a patched server is pinned, the launched binary is that build. Absent key adds no check."""
    rows = getattr(args, "speculative_rows", None)
    if rows is None:
        return []
    checks = []
    for model_id, row in speculative_rows_configuration(dict(rows), Path("/")).items():
        try:
            manifest = manifests.get(model_id)
            if manifest is None:
                raise PhysicalAdapterError("speculative rows names a model the run does not load")
            parameters = speculative_adapter_parameters(row, manifest)
            if row.patched_server_sha256 is not None and _sha256_file(args.server) != row.patched_server_sha256:
                raise PhysicalAdapterError("server binary differs from the speculative patched server pin")
        except PhysicalAdapterError as error:
            checks.append(check("speculative-rows:" + model_id, False, str(error)))
        else:
            checks.append(check(
                "speculative-rows:" + model_id, True,
                f"draft {parameters[DRAFT_SHA256_PARAMETER]} ({parameters[DRAFT_BYTES_PARAMETER]} B) "
                f"draft_max={row.draft_max} draft_min={row.draft_min} pinned={row.patched_server_sha256 is not None}",
            ))
    return checks


def _phone_probe_values(
    args: argparse.Namespace, background_values: dict[str, Any]
) -> tuple[Any, Any]:
    phone = background_values["phone-runtime"]
    if phone is None:
        phone = _retry_raw_observation(
            lambda: probe_phone_runtime_with_adb_fallback(
                args.phone_diagnostic_endpoint,
                args.phone_usb_serial,
                args.adb_port,
            ),
            lambda value: value is not None,
        )
    phone_power = background_values["phone-power"]
    if phone_power is None:
        phone_power = _retry_raw_observation(
            lambda: probe_phone_power_with_adb_fallback(
                args.phone_diagnostic_endpoint,
                args.phone_usb_serial,
                args.adb_port,
            ),
            lambda value: value is not None,
        )
    return phone, phone_power


def _phone_probe_checks(
    phone: Any, phone_power: Any
) -> tuple[PhysicalPreflightCheck, ...]:
    return (
        check(
            "phone-runtime-probe",
            phone is not None,
            "OP15 runtime telemetry is " + (
                "fresh" if phone is not None else "unavailable"
            ),
        ),
        check(
            "phone-energy-boundary",
            phone_power is not None,
            "whole-phone power telemetry is " + (
                "fresh" if phone_power is not None else "unavailable"
            ),
        ),
    )


def _normal_usb_check(
    args: argparse.Namespace,
) -> tuple[AndroidUsbRestorationReceipt | None, PhysicalPreflightCheck]:
    try:
        normal_usb_value = load_object(args.phone_normal_usb_receipt)
        normal_usb = AndroidUsbRestorationReceipt(
            serial=normal_usb_value.get("serial"),
            adb_port=normal_usb_value.get("adb_port"),
            sysfs_device=normal_usb_value.get("sysfs_device"),
            vendor_id=normal_usb_value.get("vendor_id"),
            product_id=normal_usb_value.get("product_id"),
            negotiated_speed_mbps=normal_usb_value.get(
                "negotiated_speed_mbps"
            ),
        )
        require(
            normal_usb_value == normal_usb.to_json()
            and normal_usb.serial == args.phone_usb_serial
            and normal_usb.adb_port == args.adb_port
            and normal_usb.negotiated_speed_mbps
                >= args.minimum_usb_speed_mbps,
            "normal USB receipt does not match the physical run",
        )
    except (OSError, ValueError, PhysicalAdapterError, RigPreflightError) \
            as error:
        normal_usb = None
        normal_usb_detail = "normal USB receipt failed: " + str(error)
    else:
        normal_usb_detail = (
            "pre-transition ADB receipt identifies "
            + normal_usb.serial
            + " at "
            + str(normal_usb.negotiated_speed_mbps)
            + " Mbps"
        )
    return normal_usb, check(
        "phone-identity",
        normal_usb is not None,
        normal_usb_detail,
    )


def _absent_helper_phones(args: argparse.Namespace, helpers, rows, observed) -> frozenset[str]:
    """Co-helpers that are not on the USB bus, tolerated only when elastic phones may join later.

    A phone that answers with a wrong serial or speed is present and still fails its check."""
    elastic = getattr(args, "elastic_phones", None)
    if not helpers or not elastic or not elastic.get("join"):
        return frozenset()
    by_name = {row.name: row for row in rows}
    return frozenset(
        row["device_id"] for row in helpers
        if row["device_id"] not in observed
        and not by_name["phone-usb-port:" + row["device_id"]].passed
    )


def _elastic_usb_rows(rows, observed, absent):
    """USB rows with absent co-helpers recorded, not fatal: the topology covers the present phones."""
    if not absent:
        return rows
    result = []
    for row in rows:
        device = row.name.removeprefix("phone-usb-port:")
        if device in absent:
            row = UsbTopologyCheck(row.name, True, "ABSENT_AT_START (elastic join): " + row.detail)
        elif row.name == "phone-usb-topology":
            roots = [value.root_port for value in observed.values()]
            row = UsbTopologyCheck(row.name, len(set(roots)) == len(roots),
                                   row.detail + "; absent at start: " + ",".join(sorted(absent)))
        result.append(row)
    return tuple(result)


def _helper_phone_checks(args: argparse.Namespace, manifests=None) -> list[PhysicalPreflightCheck]:
    """Bind helper USB topology, transport evidence and the campaign lifecycle."""
    helpers = [json.loads(row) for row in args.helper_phone]
    if not helpers:
        return []
    rows, observed = check_usb_topology([
        ("primary-phone", args.phone_usb_serial, None, args.minimum_usb_speed_mbps),
        *((row["device_id"], row["serial"], row["usb_sysfs_device"], row["minimum_usb_speed_mbps"])
          for row in helpers),
    ])
    absent = _absent_helper_phones(args, helpers, rows, observed)
    rows = _elastic_usb_rows(rows, observed, absent)
    paths = getattr(args, "helper_phone_evidence", ())
    if paths:
        from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import (
            load_helper_evidence, campaign_co_helper_lifecycles,
        )
        checks = [check(row.name, row.passed, row.detail) for row in rows]
        evidences = {}
        try:
            for path in paths:
                evidence = load_helper_evidence(path, server_path=args.server)
                if evidence.worker.device_id in absent:
                    # the pinned identity is verified at join time instead
                    evidences[evidence.worker.device_id] = evidence
                    checks.append(check("helper-phone-membership:" + evidence.worker.device_id, True,
                                        "ABSENT_AT_START: joins after identity-pinned preflight "
                                        + evidence.identity.identity_sha256))
                    continue
                evidence.live_preflight()
                evidences[evidence.worker.device_id] = evidence
                checks.append(check("helper-phone-transport-identity:" + evidence.worker.device_id, True,
                                    evidence.identity.identity_sha256))
                checks.append(check("helper-phone-cost-evidence:" + evidence.worker.device_id, True,
                                    "pinned kernel/link receipts and separately assumed whole-phone power"))
            require(set(evidences) == {row["device_id"] for row in helpers}, "helper evidence set differs")
            catalog = RuntimeCapabilityCatalog.from_json(load_object(args.capability_catalog))
            require(manifests is not None, "helper preflight manifests are absent")
            campaign_co_helper_lifecycles(paths, catalog, manifests, args.server)
        except (OSError, ValueError, KeyError, PhysicalAdapterError) as error:
            checks.append(check("two-phone-dispatch", False, str(error)))
        else:
            checks.append(check("two-phone-dispatch", True, "static helper lifecycle, ownership and evidence bound"))
        return checks
    return [
        *(check(row.name, row.passed, row.detail) for row in rows),
        # hardware evidence the co-helper routes need; never derived from the primary phone
        *(check("helper-phone-transport-identity:" + row["device_id"], False,
                "no qualified adb-tcp transport identity: server-token-identity, adb-forward-round-trip "
                "and scheduler-launched-session receipts are missing (two-phone gate step 3)")
          for row in helpers),
        *(check("helper-phone-cost-evidence:" + row["device_id"], False,
                "no kernel, link or power profile: the catalog withholds its co-helper routes")
          for row in helpers),
        # never run only the first phone of a multi-phone manifest
        check("two-phone-dispatch", False, "catalog, plan, ticket and adaptive policies bind co-helper "
              "phones, but the rig has no co-helper worker lifecycle (stop semantics undecided); "
              "use the two-phone mechanism gate"),
    ]


def _runtime_usb_check(
    args: argparse.Namespace,
) -> tuple[Any, PhysicalPreflightCheck]:
    runtime_usb = None
    if args.expect_functionfs_active:
        try:
            runtime_usb = probe_functionfs_usb_device(
                vendor_id=args.functionfs_vendor_id,
                product_id=args.functionfs_product_id,
            )
            require(
                runtime_usb.negotiated_speed_mbps
                    >= args.minimum_usb_speed_mbps,
                "FunctionFS USB link is below the required speed",
            )
        except (PhysicalAdapterError, RigPreflightError) as error:
            runtime_usb_detail = "runtime USB probe failed: " + str(error)
        else:
            runtime_usb_detail = (
                "active FunctionFS "
                + runtime_usb.vendor_id
                + ":"
                + runtime_usb.product_id
                + " is live at "
                + str(runtime_usb.negotiated_speed_mbps)
                + " Mbps"
            )
    else:
        runtime_usb_detail = (
            "FunctionFS verification is deferred until the scheduler-issued "
            "phone transition"
        )
    return runtime_usb, check(
        "phone-runtime-usb",
        runtime_usb is not None or not args.expect_functionfs_active,
        runtime_usb_detail,
    )


def _phone_state_checks(
    catalog: RuntimeCapabilityCatalog, phone: Any
) -> tuple[PhysicalPreflightCheck, ...]:
    phone_device_ids = {
        device.device_id
        for device in catalog.placement_profile.devices.values()
        if device.kind == "phone"
    }
    cold_phone_transition_ready = any(
        (transition.device_id in phone_device_ids
         or bool(phone_device_ids.intersection(transition.prepares_device_ids)))
        and transition.maturity == "QUALIFIED"
        for transition in catalog.transitions
    )
    return (
        check(
            "phone-memory",
            phone.available_bytes >= 768 * 1024**2,
            "OP15 available bytes are " + str(phone.available_bytes),
        ),
        check(
            "phone-thermal",
            phone.thermal_qualified,
            "OP15 temperature millic is " + str(phone.temperature_millic),
        ),
        check(
            "phone-task-server",
            phone.task_server_alive or cold_phone_transition_ready,
            (
                "phone task server alive=" + str(phone.task_server_alive)
                + "; qualified cold transition="
                + str(cold_phone_transition_ready)
            ),
        ),
    )


def _endpoint_checks(
    catalog: RuntimeCapabilityCatalog,
    phone_endpoints: tuple[str, ...],
    background_values: dict[str, Any],
) -> tuple[list[PhysicalPreflightCheck], dict[str, Any]]:
    endpoint_rows = (*catalog.executors, *catalog.composite_executors)
    live_by_endpoint = {
        endpoint: background_values["endpoint:" + endpoint]
        for endpoint in phone_endpoints
        if background_values["endpoint:" + endpoint] is not None
    }
    rig_checks = []
    for capability in endpoint_rows:
        if capability.endpoint.startswith("physical://"):
            live_by_endpoint[capability.endpoint] = probe_llama_endpoint(capability.endpoint)
            rig_checks.append(check("endpoint:" + capability.executor_id, True,
                                    "static FFN worker starts at the paid trace boundary", advisory=True))
            continue
        live = live_by_endpoint.get(capability.endpoint)
        if live is None:
            hostname = urlsplit(capability.endpoint).hostname
            if hostname in {"127.0.0.1", "localhost"}:
                live = probe_llama_endpoint(capability.endpoint)
            else:
                live = _retry_raw_observation(
                    lambda endpoint=capability.endpoint:
                        probe_llama_endpoint(endpoint),
                    lambda value: value.ready,
                )
            live_by_endpoint[capability.endpoint] = live
        port_ok, port_detail = _local_port_available(
            capability.endpoint, live
        )
        port_is_required = (
            urlsplit(capability.endpoint).hostname
                in {"127.0.0.1", "localhost"}
            or capability.executor_id == "physical:op15-phone"
        )
        session_preparation = (
            bool(getattr(capability, "phone_sessions", ()))
            and not capability.supports_whole_model
            and any(capability.device_id in row.prepares_device_ids
                    and row.maturity == "QUALIFIED" for row in catalog.transitions)
        )
        if session_preparation:
            port_is_required = False
            port_detail += "; FFN session endpoints are verified by qualified physical preparation"
        rig_checks.append(check(
            "endpoint:" + capability.executor_id,
            port_ok or not port_is_required,
            "raw endpoint probe; " + port_detail,
            advisory=not port_is_required,
        ))
    return rig_checks, live_by_endpoint


def _memory_snapshot(
    catalog: RuntimeCapabilityCatalog,
    host: Any,
    gpu: dict[str, Any],
    phone: Any,
    helpers=None,
) -> RuntimePlacementSnapshot:
    memory_pools = catalog.placement_profile.memory_pools
    phone_capacity = (
        memory_pools["op15-ram"].capacity_bytes
        if phone is None else phone.capacity_bytes
    )
    phone_available = 0 if phone is None else phone.available_bytes
    return RuntimePlacementSnapshot(
        snapshot_id="physical-preflight-memory",
        captured_at_us=0,
        valid_until_us=60_000_001,
        capacities={
            "host-ram": DeviceMemoryCapacity(
                "host-ram",
                host.memory_total_bytes,
                host.memory_total_bytes - host.memory_available_bytes,
                min(1024**3, host.memory_available_bytes),
            ),
            "cuda0-vram": DeviceMemoryCapacity(
                "cuda0-vram",
                gpu["memory_total_bytes"],
                gpu["memory_total_bytes"] - gpu["memory_free_bytes"],
                min(512 * 1024**2, gpu["memory_free_bytes"]),
            ),
            "op15-ram": DeviceMemoryCapacity(
                "op15-ram",
                phone_capacity,
                phone_capacity - phone_available,
                0,
            ),
            **{
                session.memory_resource_id: DeviceMemoryCapacity(
                    session.memory_resource_id,
                    session.resident_memory_limit_bytes,
                    0,
                    0,
                )
                for executor in catalog.executors
                for session in executor.phone_sessions
            },
            **{catalog.placement_profile.devices[device].memory_pool_id: DeviceMemoryCapacity(
                catalog.placement_profile.devices[device].memory_pool_id, sample.capacity_bytes,
                sample.capacity_bytes - sample.available_bytes, min(768 * 1024**2, sample.available_bytes))
               for device, sample in (helpers or {}).items()},
        },
    )


def _preflight_models(
    catalog: RuntimeCapabilityCatalog,
    manifests: dict[str, ModelManifest],
    model_paths: dict[str, Path],
    aliases: dict[str, str],
    first_by_model: dict[str, dict[str, Any]],
    llama_model_id: str,
    samples: dict[str, Any],
    memory: RuntimePlacementSnapshot,
    gpu: dict[str, Any],
    phone: Any,
) -> list[PhysicalPreflightModel]:
    builder = UnifiedRuntimeSnapshotBuilder(catalog)
    llama_residencies = (
        ExecutorResidencySample(
            manifests[llama_model_id], "physical:desktop-cpu", 1
        ),
    )
    phone_sample = samples.get("physical:op15-phone")
    if phone_sample is not None and phone_sample.ready:
        llama_residencies += (ExecutorResidencySample(
            manifests[llama_model_id], "physical:op15-phone", 1
        ),)
    preloaded_residencies = catalog_preloaded_residency_samples(
        catalog, manifests
    )
    telemetry = {
        "op15-phone": DeviceRuntimeTelemetry(
            temperature_millic=(
                100_000 if phone is None else phone.temperature_millic
            ),
            battery_ppm=(0 if phone is None else phone.battery_ppm),
        )
    }
    preflight_models = []
    for model_id, manifest in manifests.items():
        request = _request_for_row(first_by_model[model_id], model_id)
        snapshot = builder.build(
            snapshot_id="physical-preflight-" + model_id,
            captured_at_us=0,
            valid_until_us=60_000_001,
            memory=memory,
            executor_samples=samples,
            residencies=(
                preloaded_residencies
                + (llama_residencies if model_id == llama_model_id else ())
            ),
            device_telemetry=telemetry,
            cost_features=_profile_features(catalog, manifest, request),
            protected_work=RuntimeProtectedWorkObservation(
                observation_id="physical-preflight-idle-power",
                critical_path_end_us=request.arrival_us,
                phase_power_mw=int(gpu["power_mw"]),
                stranded_idle_power_mw=0,
                causal_tail_power_mw=0,
                sample_count=1,
                measured=True,
            ),
        )
        preflight_models.append(PhysicalPreflightModel(
            manifest,
            model_paths[model_id],
            request,
            snapshot,
            aliases[model_id],
        ))
    return preflight_models


def _result_details(
    args: argparse.Namespace,
    manifests: dict[str, ModelManifest],
    merged: list[dict[str, Any]],
    *,
    host: Any,
    gpu: dict[str, Any],
    phone: Any,
    phone_power: Any,
    phone_router_sha256: str,
    normal_usb: AndroidUsbRestorationReceipt | None,
    runtime_usb: Any,
    ffn_shard_indexes: dict[str, FfnShardIndex],
    unsupported_requests: list[dict[str, object]] = (),
) -> dict[str, object]:
    return {
        "catalog_sha256": digest(args.capability_catalog),
        "hardware": {
            "gpu": gpu,
            "host_memory_available_bytes": host.memory_available_bytes,
            "host_memory_total_bytes": host.memory_total_bytes,
            "phone": (
                None if phone is None else {
                    "available_bytes": phone.available_bytes,
                    "capacity_bytes": phone.capacity_bytes,
                    "task_server_alive": phone.task_server_alive,
                    "temperature_millic": phone.temperature_millic,
                    "thermal_qualified": phone.thermal_qualified,
                }
            ),
            "phone_power_probe_present": phone_power is not None,
            "phone_router_sha256": phone_router_sha256,
            "phone_normal_usb": (
                None if normal_usb is None else normal_usb.to_json()
            ),
            "phone_runtime_usb": (
                None if runtime_usb is None else runtime_usb.to_json()
            ),
        },
        "model_artifacts": {
            model_id: {
                "artifact_bytes": manifest.artifact_bytes,
                "artifact_sha256": manifest.artifact_sha256,
            }
            for model_id, manifest in sorted(manifests.items())
        },
        "phone_ffn_shard_indexes": {
            artifact: {
                "index_sha256": index.index_sha256,
                "shard_sha256s": [
                    row.shard_sha256 for row in index.records
                ],
            }
            for artifact, index in sorted(ffn_shard_indexes.items())
        },
        "physical_inference_executed": False,
        "preflight_schema": PREFLIGHT_RESULT_SCHEMA,
        "trace": {
            "gemma_requests": sum(item["source"] == "large" and
                                   trace_role(item["row"]) == GEMMA_ROLE for item in merged),
            "large_sha256": digest(args.large_requests),
            "llama_requests": sum(item["source"] == "overlay" for item in merged),
            "overlay_sha256": digest(args.overlay_requests),
            "qwen_requests": sum(item["source"] == "large" and
                                  trace_role(item["row"]) == QWEN_ROLE for item in merged),
            "requests": len(merged),
            "unsupported_requests": list(unsupported_requests),
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    large_rows, overlay_rows = validate_trace(
        args.large_requests,
        args.overlay_requests,
        load_object(args.trace_manifest),
    )
    expected_qwen = ModelManifest.from_json(load_object(args.qwen_manifest))
    expected_gemma = ModelManifest.from_json(load_object(args.gemma_manifest))
    llama_model_id = overlay_rows[0]["execution_model_id"]
    model_paths = {
        expected_qwen.model_id: args.qwen_model,
        expected_gemma.model_id: args.gemma_model,
        llama_model_id: args.llama_model,
    }
    catalog = RuntimeCapabilityCatalog.from_json(
        load_object(args.capability_catalog)
    )
    _require_phone_session_discovery(args, catalog)
    scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
    scheduler.register_runtime_capabilities(catalog)
    phone_endpoints = _phone_endpoints(catalog)
    background_probes = _background_probes(args, phone_endpoints)
    manifests, background_values = _register_models_under_monitor(
        args, scheduler, model_paths, background_probes
    )
    _require_manifest_identity(manifests, expected_qwen, expected_gemma)
    ffn_shard_indexes, ffn_shard_checks = _ffn_shard_preflight(
        args, expected_qwen, expected_gemma, manifests[llama_model_id]
    )
    if ffn_shard_indexes:
        scheduler.register_phone_ffn_shard_storage(_ffn_shard_storage(ffn_shard_indexes))
    aliases = artifact_aliases(catalog, manifests)
    merged = merge_rows(large_rows, overlay_rows, {
        QWEN_ROLE: expected_qwen.model_id,
        GEMMA_ROLE: expected_gemma.model_id,
    })
    if getattr(args, "replay_schedule", None) is not None:
        merged, _ = apply_named_replay_schedule(merged, load_object(args.replay_schedule))
    first_by_model = {}
    for item in merged:
        first_by_model.setdefault(item["model_id"], item["row"])
    require(set(first_by_model) == set(manifests), "three-model trace coverage")

    phone_router_sha256, phone_router_detail = _phone_router_identity(args)

    rig_checks = _model_checks(
        catalog, manifests, expected_qwen, expected_gemma, llama_model_id
    )
    rig_checks.extend(ffn_shard_checks)
    host = LinuxHostRuntimeProbe().sample()
    gpu = nvidia_gpu_snapshot()
    rig_checks.extend(_gpu_checks(gpu))
    rig_checks.append(_rapl_check())
    rig_checks.extend(_device_power_checks(args))
    rig_checks.extend(_speculative_rows_checks(args, manifests))
    rig_checks.append(check(
        "phone-router-binary",
        True,
        phone_router_detail,
    ))

    phone, phone_power = _phone_probe_values(args, background_values)
    rig_checks.extend(_phone_probe_checks(phone, phone_power))
    normal_usb, normal_usb_check = _normal_usb_check(args)
    rig_checks.append(normal_usb_check)
    runtime_usb, runtime_usb_check = _runtime_usb_check(args)
    rig_checks.append(runtime_usb_check)
    rig_checks.extend(_helper_phone_checks(args, manifests))
    if phone is not None:
        rig_checks.extend(_phone_state_checks(catalog, phone))

    endpoint_checks, live_by_endpoint = _endpoint_checks(
        catalog, phone_endpoints, background_values
    )
    rig_checks.extend(endpoint_checks)
    samples = materialize_preflight_executor_samples(
        catalog,
        live_by_endpoint,
        bootstrap_executor_ids=("physical:desktop-cpu",),
    )

    from research_dev.scheduler.adapters.probes import probe_android_phone_runtime
    helper_runtime = {}
    absent_helpers = frozenset(
        row.check_id.removeprefix("helper-phone-membership:") for row in rig_checks
        if row.check_id.startswith("helper-phone-membership:")
    )
    for raw in getattr(args, "helper_phone", ()):
        row = json.loads(raw)
        if row["device_id"] in absent_helpers:
            continue
        sample = probe_android_phone_runtime(row["serial"], row["adb_port"])
        require(sample is not None, "helper phone runtime telemetry is unavailable: " + row["device_id"])
        helper_runtime[row["device_id"]] = sample
    memory = _memory_snapshot(catalog, host, gpu, phone, helper_runtime)
    preflight_models = _preflight_models(
        catalog,
        manifests,
        model_paths,
        aliases,
        first_by_model,
        llama_model_id,
        samples,
        memory,
        gpu,
        phone,
    )

    report = run_physical_preflight(
        scheduler,
        catalog,
        tuple(preflight_models),
        executable_paths={
            "bridge": args.bridge,
            "resident-server": args.resident_server,
            "server": args.server,
            **(
                {}
                if args.phone_router is None
                else {"phone-router": args.phone_router}
            ),
        },
        required_paths={
            "close-helper": args.close_helper,
            "large-trace": args.large_requests,
            "overlay-trace": args.overlay_requests,
            **{
                "model:" + model_id: path
                for model_id, path in model_paths.items()
            },
        },
        library_directories={
            "cuda": args.cuda_lib_dir,
            "resident": args.resident_lib_dir,
        },
        rig_checks=tuple(rig_checks),
        trace_requests=tuple(
            (_trace_request_for_row(item["row"], item["model_id"]), item["model_id"])
            for item in merged
        ),
    )
    result = report.to_json()
    result.update(_result_details(
        args,
        manifests,
        merged,
        host=host,
        gpu=gpu,
        phone=phone,
        phone_power=phone_power,
        phone_router_sha256=phone_router_sha256,
        normal_usb=normal_usb,
        runtime_usb=runtime_usb,
        ffn_shard_indexes=ffn_shard_indexes,
        unsupported_requests=[
            dict(row) for row in report.request_shapes
            if not row.get("supported", True)
        ],
    ))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-requests", type=Path, required=True)
    parser.add_argument("--overlay-requests", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--replay-schedule", type=Path)
    parser.add_argument("--capability-catalog", type=Path, required=True)
    parser.add_argument(
        "--observation-store-input", type=Path, required=True
    )
    parser.add_argument(
        "--adaptive-observation-store-input", type=Path, required=True
    )
    parser.add_argument("--qwen-manifest", type=Path, required=True)
    parser.add_argument("--gemma-manifest", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--gemma-model", type=Path, required=True)
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--resident-server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--resident-lib-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--qwen-ffn-shards")
    parser.add_argument("--gemma-ffn-shards")
    parser.add_argument("--llama-ffn-shards")
    parser.add_argument("--phone-router", type=Path)
    parser.add_argument(
        "--phone-router-receipt", type=Path
    )
    parser.add_argument("--phone-session-discovery", type=Path)
    parser.add_argument("--close-helper", type=Path, required=True)
    parser.add_argument("--phone-diagnostic-endpoint", required=True)
    parser.add_argument("--phone-battery-ppm", type=int, required=True)
    parser.add_argument("--phone-usb-serial", required=True)
    parser.add_argument(
        "--phone-normal-usb-receipt", type=Path, required=True
    )
    parser.add_argument("--functionfs-vendor-id", default="18d1")
    parser.add_argument("--functionfs-product-id", default="2d00")
    parser.add_argument("--expect-functionfs-active", action="store_true")
    parser.add_argument("--adb-port", type=int, required=True)
    parser.add_argument("--minimum-usb-speed-mbps", type=int, default=5000)
    parser.add_argument("--helper-phone", action="append", default=[],
                        help="JSON row of one rig helper phone (launch.py)")
    parser.add_argument("--helper-phone-evidence", type=Path, action="append", default=[])
    parser.add_argument("--elastic-phones-json", dest="elastic_phones", type=elastic_phones_json,
                        help="campaign elastic_phones: an absent co-helper is recorded when it may join")
    parser.add_argument("--device-power-json", dest="device_power", type=device_power_json,
                        help="campaign device_power: the sudoers capability is probed (advisory)")
    parser.add_argument("--speculative-rows-json", dest="speculative_rows", type=speculative_rows_json,
                        help="campaign speculative_rows: the draft GGUF, its vocabulary and the server pin are checked")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists(), "new output")
    require(0 <= args.phone_battery_ppm <= 1_000_000, "phone battery ppm")
    require(
        (args.phone_router is None) == (args.phone_router_receipt is None),
        "legacy phone router and receipt must be supplied together",
    )
    return args


def failure_report(
    args: argparse.Namespace, error: BaseException
) -> dict[str, object]:
    checks = []
    regular_files = {
        "adaptive-observation-store-input": (
            args.adaptive_observation_store_input
        ),
        "capability-catalog": args.capability_catalog,
        "close-helper": args.close_helper,
        "gemma-manifest": args.gemma_manifest,
        "gemma-model": args.gemma_model,
        "large-requests": args.large_requests,
        "llama-model": args.llama_model,
        "overlay-requests": args.overlay_requests,
        "observation-store-input": args.observation_store_input,
        "phone-normal-usb-receipt": args.phone_normal_usb_receipt,
        "qwen-manifest": args.qwen_manifest,
        "qwen-model": args.qwen_model,
        "trace-manifest": args.trace_manifest,
    }
    if args.phone_session_discovery is not None:
        regular_files["phone-session-discovery"] = (
            args.phone_session_discovery
        )
    for name, spec in (
        ("qwen-ffn-shards", args.qwen_ffn_shards),
        ("gemma-ffn-shards", args.gemma_ffn_shards),
        ("llama-ffn-shards", args.llama_ffn_shards),
    ):
        if spec is not None:
            local, separator, _ = spec.rpartition("=")
            if separator and local:
                regular_files[name] = Path(local)
    executables = {
        "adb": args.adb,
        "bridge": args.bridge,
        "resident-server": args.resident_server,
        "server": args.server,
        **(
            {}
            if args.phone_router is None
            else {"phone-router": args.phone_router}
        ),
    }
    if args.phone_router_receipt is not None:
        regular_files["phone-router-receipt"] = args.phone_router_receipt
    directories = {
        "cuda-lib-dir": args.cuda_lib_dir,
        "resident-lib-dir": args.resident_lib_dir,
    }
    for name, path in sorted(regular_files.items()):
        checks.append(check(
            "path:" + name,
            path.is_file(),
            "required file " + str(path),
        ).to_json())
    for name, path in sorted(executables.items()):
        checks.append(check(
            "executable:" + name,
            path.is_file() and os.access(path, os.X_OK),
            "required executable " + str(path),
        ).to_json())
    for name, path in sorted(directories.items()):
        checks.append(check(
            "directory:" + name,
            path.is_dir(),
            "required directory " + str(path),
        ).to_json())
    try:
        gpu_probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_rows = tuple(
            tuple(field.strip() for field in line.split(","))
            for line in gpu_probe.stdout.splitlines()
            if line.strip()
        )
        gpu_ok = any(
            len(row) == 3
            and row[0] == EXPECTED_GPU_NAME
            and row[1] == EXPECTED_GPU_UUID
            and int(row[2]) * 1024 * 1024
                == EXPECTED_GPU_CAPACITY_BYTES
            for row in gpu_rows
        )
        gpu_detail = "discovered GPUs: " + "; ".join(
            ",".join(row) for row in gpu_rows
        )
    except BaseException as probe_error:
        gpu_ok = False
        gpu_detail = "GPU probe failed: " + str(probe_error)
    checks.append(check(
        "gpu-identity",
        gpu_ok,
        gpu_detail,
    ).to_json())
    try:
        rapl = rapl_package_snapshot()
    except BaseException as probe_error:
        rapl = None
        rapl_detail = "RAPL probe failed: " + str(probe_error)
    else:
        rapl_detail = "RAPL package energy counter is readable"
    checks.append(check(
        "cpu-energy-boundary", rapl is not None, rapl_detail
    ).to_json())
    phone = probe_phone_runtime_with_adb_fallback(
        args.phone_diagnostic_endpoint,
        args.phone_usb_serial,
        args.adb_port,
    )
    phone_power = probe_phone_power_with_adb_fallback(
        args.phone_diagnostic_endpoint,
        args.phone_usb_serial,
        args.adb_port,
    )
    checks.append(check(
        "phone-runtime-probe",
        phone is not None,
        "OP15 runtime telemetry is " + (
            "fresh" if phone is not None else "unavailable"
        ),
    ).to_json())
    checks.append(check(
        "phone-energy-boundary",
        phone_power is not None,
        "whole-phone power telemetry is " + (
            "fresh" if phone_power is not None else "unavailable"
        ),
    ).to_json())
    return {
        "blocker": type(error).__name__ + ": " + str(error),
        "checks": sorted(checks, key=lambda row: row["check_id"]),
        "desktop_baseline_ready": False,
        "phone_assistance_ready": False,
        "phone_route_reporting_blocker": (
            "catalog and model artifacts must pass before route evidence "
            "can be enumerated"
        ),
        "physical_inference_executed": False,
        "preflight_schema": PREFLIGHT_RESULT_SCHEMA,
        "shadow_or_unavailable_phone_routes": [],
        "status": "BLOCKED",
    }


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except BaseException as error:
        result = failure_report(args, error)
    args.output.write_bytes(canonical(result))
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
    }, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
