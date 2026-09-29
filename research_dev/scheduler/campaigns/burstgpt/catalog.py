#!/usr/bin/env python3
"""Materialize the measured F16 BurstGPT runtime capability catalog."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DATA_DIR = HERE / "data"
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler.campaigns.burstgpt import (  # noqa: E402
    admission_profile as admission,
)
from research_dev.scheduler.campaigns.burstgpt.overlay_catalog import (  # noqa: E402
    placement_profile,
)
from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import (  # noqa: E402
    load_helper_evidence, extend_helper_profile, extend_helper_overlay, validate_helper_declaration,
)
from research_dev.scheduler.config import (  # noqa: E402
    CampaignModelConfiguration,
    load_scheduler_configuration,
)
from research_dev.scheduler import (  # noqa: E402
    load_cached_gguf_manifest,
    ModelManifest,
    RuntimeCapabilityCatalog,
    RuntimeDesktopControlProfile,
    RuntimePhonePowerProfile,
    RuntimeRouteShapeProfile,
)
from research_dev.scheduler._internal.plan_contracts.co_helpers import (  # noqa: E402
    RuntimeCoHelperDeclaration,
    RuntimeCoHelperPhone,
)
from research_dev.scheduler._internal.plan_contracts.common import RuntimePlanError  # noqa: E402
from research_dev.scheduler.adapters.ffn_shards import FfnShardIndex, FfnShardIndexError  # noqa: E402
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.adapters.speculative_rows import (  # noqa: E402
    PATCHED_SERVER_PARAMETER,
    ROW_BUDGET_PARAMETER,
    per_request_draft_max,
    speculative_adapter_parameters,
)
from research_dev.scheduler.adapters import (  # noqa: E402
    add_phone_session_memory_pools,
    apply_assumed_phone_power_profile,
    RuntimeModelEndpointCapability,
    RuntimePhysicalTopology,
    RuntimeHelperPhoneTopology,
    base_executor_capabilities,
    derived_desktop_gpu_first_layer,
    load_transport_qualification_identity,
    materialize_measured_usb_links,
    materialize_desktop_control,
    materialize_cpu_phone_endpoints,
    materialize_whole_model_endpoint,
    measured_desktop_control_profile,
    measured_ffn_assist_profile,
    merge_runtime_capability_catalogs,
    model_composite_capabilities,
    model_transition_capabilities,
    phone_sessions_from_json,
    resource_profiles_for_catalog,
)


class MaterializationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def measured_desktop_plans(
    path: Path,
    manifests: dict[str, ModelManifest],
) -> dict[str, dict[str, Any]]:
    value = load(path)
    require(
        value.get("schema") == "s42-measured-desktop-baseline-plans-v1"
        and type(value.get("plans")) is list,
        "desktop baseline plan schema",
    )
    rows: dict[str, dict[str, Any]] = {}
    block_counts = {
        manifest.artifact_sha256: manifest.block_count
        for manifest in manifests.values()
    }
    for row in value["plans"]:
        require(type(row) is dict, "desktop baseline plan row")
        artifact = row.get("artifact_sha256")
        first_layer = row.get("gpu_first_layer")
        evidence_ids = row.get("evidence_ids")
        identity_evidence_id = row.get("identity_evidence_id")
        adapter_parameters = row.get("adapter_parameters", {})
        require(
            type(artifact) is str
            and artifact in block_counts
            and artifact not in rows
            and type(first_layer) is int
            and 0 <= first_layer <= block_counts[artifact]
            and row.get("maturity") == "QUALIFIED"
            and type(evidence_ids) is list
            and evidence_ids
            and all(
                type(evidence_id) is str
                and evidence_id.startswith("sha256:")
                and len(evidence_id) == 71
                for evidence_id in evidence_ids
            ),
            "desktop baseline plan identity",
        )
        require(
            type(row.get("desktop_control_profile_id")) is str
            and bool(row["desktop_control_profile_id"])
            and row["desktop_control_profile_id"].isascii()
            and type(row.get("desktop_executor_id")) is str
            and bool(row["desktop_executor_id"])
            and row["desktop_executor_id"].isascii()
            and type(row.get("desktop_placement_sha256")) is str
            and row["desktop_placement_sha256"].startswith("sha256:")
            and len(row["desktop_placement_sha256"]) == 71
            and type(identity_evidence_id) is str
            and identity_evidence_id in evidence_ids,
            "desktop baseline physical profile binding",
        )
        require(
            type(adapter_parameters) is dict
            and all(
                type(name) is str
                and bool(name)
                and name.isascii()
                and (
                    type(parameter) is int
                    and parameter >= 0
                    or type(parameter) is str
                    and bool(parameter)
                    and parameter.isascii()
                )
                for name, parameter in adapter_parameters.items()
            )
            and not {
                "context_resource_id",
                "cpu_device_id",
                "gpu_device_id",
                "gpu_layers",
                "request_memory_mode",
            }.intersection(adapter_parameters),
            "desktop baseline adapter parameters",
        )
        require(
            adapter_parameters.get("cuda_graph_mode", "default")
                in ("default", "disabled")
            and row.get("cuda_graph_mode", "default")
                == adapter_parameters.get("cuda_graph_mode", "default"),
            "desktop baseline CUDA graph mode identity",
        )
        capacity_keys = {
            "capacity_parent_calibration_live_free_bytes",
            "capacity_parent_maximum_gpu_layers",
            "capacity_parent_peak_vram_bytes",
            "capacity_parent_qualification_sha256",
            "capacity_parent_required_with_reserve_bytes",
            "capacity_parent_source_placement_sha256",
        }
        present_capacity_keys = capacity_keys.intersection(adapter_parameters)
        if present_capacity_keys:
            require(
                present_capacity_keys == capacity_keys
                and type(adapter_parameters[
                    "capacity_parent_maximum_gpu_layers"
                ]) is int
                and adapter_parameters[
                    "capacity_parent_maximum_gpu_layers"
                ] >= block_counts[artifact] - first_layer
                and all(
                    type(adapter_parameters[name]) is int
                    and adapter_parameters[name] > 0
                    for name in (
                        "capacity_parent_calibration_live_free_bytes",
                        "capacity_parent_peak_vram_bytes",
                        "capacity_parent_required_with_reserve_bytes",
                    )
                )
                and all(
                    type(adapter_parameters[name]) is str
                    and adapter_parameters[name].startswith("sha256:")
                    and len(adapter_parameters[name]) == 71
                    for name in (
                        "capacity_parent_qualification_sha256",
                        "capacity_parent_source_placement_sha256",
                    )
                )
                and adapter_parameters[
                    "capacity_parent_qualification_sha256"
                ] in evidence_ids,
                "capacity-aware desktop parent evidence",
            )
        rows[artifact] = row
    require(set(rows) == set(block_counts), "desktop baseline plan coverage")
    return rows


def validate_desktop_control_bindings(
    plans: dict[str, dict[str, Any]],
    controls: tuple[RuntimeDesktopControlProfile, ...],
) -> None:
    by_artifact = {
        control.artifact_sha256: control for control in controls
    }
    require(
        set(by_artifact) == set(plans),
        "desktop control binding coverage",
    )
    for artifact, plan in plans.items():
        control = by_artifact[artifact]
        require(
            control.profile_id == plan["desktop_control_profile_id"]
            and control.executor_id == plan["desktop_executor_id"]
            and control.placement_sha256
                == plan["desktop_placement_sha256"]
            and control.cuda_graph_mode == plan.get("cuda_graph_mode", "default")
            and plan["identity_evidence_id"] in control.evidence_ids,
            "desktop control differs from physical profile binding",
        )


def validate_final_catalog(
    catalog: RuntimeCapabilityCatalog,
    large_manifests: tuple[ModelManifest, ...],
    overlay_manifest: ModelManifest | None,
    *,
    require_whole_phone: bool,
) -> None:
    devices = catalog.placement_profile.devices
    controls = {
        row.artifact_sha256: row
        for row in catalog.desktop_control_profiles
    }
    require(
        devices[catalog.fallback.device_id].kind == "cpu",
        "qualified recovery fallback must remain CPU",
    )
    for manifest in large_manifests:
        control = controls.get(manifest.artifact_sha256)
        require(
            control is not None
            and control.maturity == "QUALIFIED"
            and any(
                devices[row.primary_device_id].kind == "gpu"
                for row in control.operator_placements
            ),
            "large-model desktop control must be qualified GPU or GPU+CPU",
        )
        assisted = tuple(
            row for row in catalog.composite_executors
            if row.artifact_sha256 == manifest.artifact_sha256
            and row.helper_device_id is not None
        )
        require(
            assisted
            and all(row.baseline_executor_id is not None for row in assisted),
            "assisted route must retain an exact desktop parent",
        )
    phone_sessions = {
        session.session_id: session
        for executor in catalog.executors
        for session in executor.phone_sessions
    }
    if phone_sessions:
        require(
            all(session.ready for session in phone_sessions.values()),
            "catalog contains an unavailable discovered phone session",
        )
    if overlay_manifest is not None:
        control = controls.get(overlay_manifest.artifact_sha256)
        require(
            control is not None
            and any(
                devices[row.primary_device_id].kind == "gpu"
                for row in control.operator_placements
            ),
            "overlay desktop control must use the GPU",
        )
        if require_whole_phone:
            whole_phone_executor_ids = {
                row.executor_id
                for row in catalog.transitions
                if row.artifact_sha256
                    == overlay_manifest.artifact_sha256
                and row.executor_id is not None
                and devices[row.device_id].kind == "phone"
            }
            require(
                any(
                    row.executor_id in whole_phone_executor_ids
                    and devices[row.device_id].kind == "phone"
                    and row.supports_whole_model
                    and row.adapter_parameters.get("model_alias")
                        == overlay_manifest.model_id
                    for row in catalog.executors
                ),
                "overlay whole-phone route is not executable",
            )


def physical_result(
    calibration_dir: Path,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    result_path = calibration_dir / f"{name}.RESULT.json"
    phone_path = calibration_dir / f"{name}.PHONE_ENERGY_V3.json"
    result = load(result_path)
    phone = load(phone_path)
    require(
        result.get("status") == "PASS"
        and result.get("metrics", {}).get("completed") == 74
        and phone.get("status") == "PASS"
        and phone.get("boundary") == "paid_trace_interval"
        and math.isclose(
            result["metrics"]["duration_s"],
            phone["duration_s"],
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ),
        f"physical result identity: {name}",
    )
    return (
        result,
        phone,
        ("sha256:" + digest(result_path), "sha256:" + digest(phone_path)),
    )


def latency_model(
    train: dict[str, Any],
    holdout: dict[str, Any],
    role: str,
) -> tuple[dict[str, Any], dict[str, tuple[int, int]]]:
    train_rows = admission.observations(train, role, "train")
    holdout_rows = admission.observations(holdout, role, "holdout")
    latency, _ = admission.fit_role(train_rows, holdout_rows)
    all_rows = train_rows + holdout_rows
    ranges = {
        feature: (
            min(row["features"][feature] for row in all_rows),
            max(row["features"][feature] for row in all_rows),
        )
        for feature in admission.FEATURES
        if feature not in {"input_tokens", "output_tokens"}
    }
    return latency, ranges


def route_profiles(
    model_configs: tuple[CampaignModelConfiguration, ...],
    manifests: dict[str, ModelManifest],
    calibration: dict[str, list[tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]]],
    device_ids: tuple[str, str, str],
    desktop_arm: str,
    assisted_arm: str,
) -> tuple[RuntimeRouteShapeProfile, ...]:
    rows = []
    for config in model_configs:
        role = config.model_key
        manifest = manifests[role]
        assert config.calibration_key is not None
        phone_profile = measured_ffn_assist_profile(
            manifest,
            tuple(
                result["fp16_resident_scheduler"]["phase_decisions"][
                    config.calibration_key
                ]
                for result, _, _ in calibration[assisted_arm]
            ),
        )
        for arm, executor_id, route_family, assisted, axis, fraction in (
            (
                desktop_arm,
                f"physical:{role}:desktop",
                "layer_placement",
                None,
                "none",
                0,
            ),
            (
                assisted_arm,
                (
                    f"physical:{role}:phone-assisted:"
                    + phone_profile.route_family
                ),
                phone_profile.route_family,
                phone_profile.operator_kind,
                phone_profile.split_axis,
                phone_profile.split_fraction_ppm,
            ),
        ):
            arm_rows = calibration[arm]
            train, holdout = arm_rows[0][0], arm_rows[1][0]
            latency, feature_ranges = latency_model(
                train, holdout, role
            )
            observations = (
                admission.observations(train, role, "train")
                + admission.observations(holdout, role, "holdout")
            )
            input_values = [row["features"]["input_tokens"] for row in observations]
            output_values = [row["features"]["output_tokens"] for row in observations]
            model = latency["cost_us"]
            coefficients = dict(model["coefficients"])
            input_coefficient = coefficients.pop("input_tokens")
            output_coefficient = coefficients.pop("output_tokens")
            evidence = tuple(sorted({
                evidence_id
                for _, _, evidence_ids in arm_rows
                for evidence_id in evidence_ids
            }))
            devices = device_ids[:2] if arm == desktop_arm else device_ids
            for residency in ("cold", "hot", "warm"):
                rows.append(RuntimeRouteShapeProfile(
                    selector_id=(
                        f"physical:{role}:{arm}:{residency}:v1"
                    ),
                    artifact_sha256=manifest.artifact_sha256,
                    route_family=route_family,
                    device_ids=devices,
                    assisted_operator_kind=assisted,
                    split_axis=axis,
                    split_fraction_ppm=fraction,
                    residency_variant=residency,
                    minimum_input_tokens=min(input_values),
                    maximum_input_tokens=max(input_values),
                    minimum_output_tokens=min(output_values),
                    maximum_output_tokens=max(output_values),
                    service_fixed_us=model["fixed"],
                    service_input_token_us=input_coefficient,
                    service_output_token_us=output_coefficient,
                    service_upper_add_us=latency["ucb_add_us"],
                    energy_fixed_uj=0,
                    energy_input_token_uj=0,
                    energy_output_token_uj=0,
                    energy_lower_error_ppm=0,
                    energy_upper_error_ppm=0,
                    sample_count=2,
                    maturity="SHADOW",
                    evidence_ids=evidence,
                    feature_ranges=feature_ranges,
                    service_feature_coefficients_us=coefficients,
                    energy_feature_coefficients_uj={},
                    executor_id=executor_id,
                ))
    return tuple(rows)


def write_new(path: Path, value: object) -> None:
    require(path.is_absolute() and not path.exists(), "output must be new")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), "temporary output exists")
    try:
        temporary.write_bytes(canonical(value))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _calibration_results(evidence) -> dict[str, list]:
    return {
        arm: [
            physical_result(evidence.calibration_directory, name)
            for name in run_names
        ]
        for arm, run_names in evidence.calibration_runs.items()
    }


def _measured_placement_profile(
    measured: dict[str, Any],
    memory_bytes: dict[str, int],
    evidence,
    topology_config,
):
    profile = placement_profile(measured, memory_bytes)
    return replace(
        profile,
        devices={
            device_id: replace(
                device,
                allocation_limit_bytes=memory_bytes[device_id],
            )
            for device_id, device in profile.devices.items()
        },
        links=(
            tuple(
                row for row in profile.links
                if not any(
                    profile.devices[device_id].kind.startswith("phone")
                    for device_id in (
                        row.source_device, row.target_device
                    )
                )
            )
            + materialize_measured_usb_links(
                evidence.transport_qualification_directories,
                load_transport_qualification_identity(
                    evidence.transport_qualification_identity_path
                ),
                host_device_id=topology_config.cpu_device_id,
                phone_device_id=topology_config.phone_device_id,
            )
        ),
    )


def _apply_phone_power(profile, power, topology_config):
    phone_power_profiles = ()
    if power.allow_assumed_for_scheduling:
        require(
            power.evidence_kind == "ASSUMED_4P5W"
            and power.active_power_mw == 4_500
            and power.idle_power_mw == 875,
            "assumed phone power profile",
        )
        phone_domains = {
            row.kernel.domain_id
            for row in profile.kernels.values()
            if row.device_id == topology_config.phone_device_id
        }
        require(
            len(phone_domains) == 1,
            "phone power energy domain",
        )
        phone_power = RuntimePhonePowerProfile.assumed_4p5w(
            device_id=topology_config.phone_device_id,
            domain_id=next(iter(phone_domains)),
            allow_assumed_for_scheduling=(
                power.allow_assumed_for_scheduling
            ),
            minimum_battery_ppm=power.minimum_battery_ppm,
        )
        profile = apply_assumed_phone_power_profile(
            profile, phone_power
        )
        phone_power_profiles = (phone_power,)
    return profile, phone_power_profiles


def _physical_topology(rig, topology_config, phone_sessions, helper_device_ids=()) -> RuntimePhysicalTopology:
    resource_capacities = {
        row.resource_id: row.capacity for row in rig.resources
    }
    resource_identities = {
        row.resource_id: row.identity for row in rig.resources
    }
    return RuntimePhysicalTopology(
        cpu_device_id=topology_config.cpu_device_id,
        gpu_device_id=topology_config.gpu_device_id,
        phone_device_id=topology_config.phone_device_id,
        cpu_resource_id=topology_config.cpu_resource_id,
        gpu_resource_id=topology_config.gpu_resource_id,
        host_memory_resource_id=topology_config.host_memory_resource_id,
        gpu_memory_resource_id=topology_config.gpu_memory_resource_id,
        phone_memory_resource_id=topology_config.phone_memory_resource_id,
        functionfs_resource_id=topology_config.functionfs_resource_id,
        phone_transport_resource_ids=(
            topology_config.phone_transport_resource_ids
        ),
        phone_compute_resource_ids=(
            topology_config.phone_compute_resource_ids
        ),
        gpu_exclusive_residency_resource_id=(
            topology_config.gpu_exclusive_residency_resource_id
        ),
        resource_capacities=resource_capacities,
        resource_identities=resource_identities,
        phone_sessions=phone_sessions,
        helper_phones=tuple(RuntimeHelperPhoneTopology(**row.to_json())
                            for row in topology_config.helper_phones if row.device_id in helper_device_ids),
        phone_exclusive_residency_resource_id=(
            topology_config.phone_exclusive_residency_resource_id
        ),
    )


def _speculative_parameters(
    configuration,
    manifest: ModelManifest,
    config: CampaignModelConfiguration,
    desktop_plan: dict[str, Any],
) -> dict[str, int | str]:
    """Adapter parameters of one model's campaign ``speculative_rows`` entry (empty without one).

    The draft is bound by digest here so every launch and route identity carries it. Under a
    patched-server pin the phone route must still leave a lone request a draft row within the
    phone contract (``min(ubatch, parallel)`` or the declared ``ffn_max_tokens``); the stock
    server drafts on desktop-only launches alone, so nothing else is checked for it."""
    if configuration is None:
        return {}
    try:
        parameters = speculative_adapter_parameters(configuration, manifest)
        if PATCHED_SERVER_PARAMETER in parameters:
            launch = {**dict(config.runtime_parameters), **desktop_plan.get("adapter_parameters", {})}
            parallel, ubatch = launch.get("parallel"), launch.get("ubatch_size")
            require(type(parallel) is int and parallel > 0 and type(ubatch) is int and ubatch > 0,
                    f"speculative rows of {manifest.model_id} need the desktop plan's parallel and ubatch")
            max_tokens = config.phone_adapter_parameters.get("ffn_max_tokens", min(ubatch, parallel))
            budget = min(parameters.get(ROW_BUDGET_PARAMETER, max_tokens), max_tokens)
            require(per_request_draft_max(draft_max=configuration.draft_max, row_budget=budget,
                                          parallel=parallel, reserved_rows=()) > 0,
                    f"speculative rows of {manifest.model_id}: {parallel} decode slots fill the "
                    f"{budget}-row phone budget, a lone request gets no draft row")
    except PhysicalAdapterError as error:
        raise MaterializationError(f"speculative rows of {manifest.model_id}: {error}") from error
    return parameters


def _model_endpoint(
    role: str,
    manifest: ModelManifest,
    config: CampaignModelConfiguration,
    endpoints,
    desktop_plans: dict[str, dict[str, Any]],
    arm_evidence: dict[str, tuple[str, ...]],
    evidence,
    speculative_parameters: dict[str, int | str] | None = None,
) -> RuntimeModelEndpointCapability:
    cpu_endpoint_id = config.endpoint_ids.get("cpu")
    desktop_plan = desktop_plans[manifest.artifact_sha256]
    capacity_qualified = (
        "capacity_parent_qualification_sha256"
        in desktop_plan.get("adapter_parameters", {})
    )
    return RuntimeModelEndpointCapability(
        manifest=manifest,
        desktop_executor_id=f"physical:{role}:desktop",
        desktop_endpoint=endpoints[config.endpoint_ids["desktop"]],
        desktop_backend=config.backend_ids["desktop"],
        phone_executor_prefix=f"physical:{role}:phone-assisted",
        phone_endpoint=endpoints[config.endpoint_ids["phone"]],
        phone_backend=config.backend_ids["phone"],
        desktop_gpu_first_layer=desktop_plan["gpu_first_layer"],
        adapter_parameters={
            **dict(config.runtime_parameters),
            **desktop_plan.get("adapter_parameters", {}),
            **(speculative_parameters or {}),
        },
        phone_adapter_parameters=dict(
            config.phone_adapter_parameters
        ),
        phone_runtime_control_protocol="decode-boundary-v1",
        desktop_evidence_ids=(
            tuple(desktop_plan["evidence_ids"])
            if capacity_qualified else
            arm_evidence[evidence.desktop_calibration_arm]
        ),
        phone_evidence_ids=arm_evidence[
            evidence.assisted_calibration_arm
        ],
        phone_preloaded=config.phone_residency_mode == "preloaded",
        phone_resident_limit_bytes=config.phone_resident_limit_bytes,
        ffn_max_runtime_partitions=config.ffn_max_runtime_partitions,
        phone_batch_plans=config.phone_batch_plans,
        qualified_phone_batch_plans=(
            config.qualified_phone_batch_plans
        ),
        cpu_executor_id=(
            None if cpu_endpoint_id is None
            else f"physical:{role}:cpu"
        ),
        cpu_endpoint=(
            None if cpu_endpoint_id is None
            else endpoints[cpu_endpoint_id]
        ),
        cpu_backend=(
            None if cpu_endpoint_id is None
            else config.backend_ids["cpu"]
        ),
        cpu_phone_executor_prefix=(
            None if cpu_endpoint_id is None
            else f"physical:{role}:cpu-phone-assisted"
        ),
        cpu_phone_endpoint=(
            None if cpu_endpoint_id is None
            else endpoints[cpu_endpoint_id]
        ),
        cpu_phone_backend=(
            None if cpu_endpoint_id is None
            else config.backend_ids.get(
                "cpu_phone", config.backend_ids["cpu"]
            )
        ),
        cpu_adapter_parameters=(dict(config.cpu_runtime_parameters) if config.cpu_runtime_parameters else None),
        cpu_evidence_ids=config.cpu_evidence_ids or None,
    )


def helper_phone_co_helpers(rig, config, manifest: ModelManifest) -> RuntimeCoHelperDeclaration | None:
    """Static FFN co-helper phones of one model: rig helper phones plus their shard indexes.

    Helper order is the rig order (it fixes each helper's request-id range). A co-helper serves
    exactly one full-width shard of this model behind a fixed host forward port.
    """
    if not config.helper_phone_ffn_shards:
        return None
    helpers = []
    try:
        for row in rig.helper_phones:
            if row.device_id not in config.helper_phone_ffn_shards:
                continue
            index = FfnShardIndex.load(*config.helper_phone_ffn_shards[row.device_id])
            require(
                index.parent_sha256 == manifest.artifact_sha256
                and len(index.records) == 1
                and index.records[0].columns == manifest.feed_forward_length,
                f"helper phone {row.device_id} needs one full-width shard of {manifest.model_id}",
            )
            require(row.forward_port > 0, f"helper phone {row.device_id} needs a fixed host forward port")
            record = index.records[0]
            label = row.device_id.split("-", 1)[0]
            helpers.append(RuntimeCoHelperPhone(
                device_id=row.device_id, serial=row.serial, label=label,
                session_id=label.upper() + "0", layer_mask=record.layer_mask,
                column_quantum=row.column_quantum, max_tokens=row.max_tokens,
                shard_sha256=record.shard_sha256, resident_bytes=record.shard_bytes,
                transport_parameters={
                    "adb_port": row.adb_port, "adb_serial": row.serial, "ffn_transport": "adb-tcp",
                    "ffn_worker_host": "127.0.0.1", "ffn_worker_port": row.forward_port,
                    "phone_worker_port": row.worker_port,
                },
            ))
        declaration = RuntimeCoHelperDeclaration(
            primary_label=rig.topology.phone_device_id.split("-", 1)[0],
            primary_serial=rig.phone.serial,
            helpers=tuple(helpers),
        )
        if config.phone_ffn_shard_index_path is not None:
            # per-model disjointness: the primary phone's stored slices never reach a co-helper layer
            primary = FfnShardIndex.load(config.phone_ffn_shard_index_path, config.phone_ffn_shard_directory)
            require(
                not any(row.layer_mask & declaration.layer_mask for row in primary.records),
                f"primary phone shards of {manifest.model_id} overlap co-helper layers",
            )
        return declaration
    except (FfnShardIndexError, RuntimePlanError) as error:
        raise MaterializationError(f"helper phone co-helper of {manifest.model_id}: {error}") from error


def _desktop_controls(
    model_endpoints: dict[str, RuntimeModelEndpointCapability],
    composites: tuple,
    desktop_plans: dict[str, dict[str, Any]],
) -> tuple[RuntimeDesktopControlProfile, ...]:
    return tuple(
        measured_desktop_control_profile(
            model_endpoints[role].manifest,
            executor_id=model_endpoints[role].desktop_executor_id,
            operator_placements=next(
                row.operator_placements for row in composites
                if row.executor_id
                    == model_endpoints[role].desktop_executor_id
            ),
            evidence_ids=desktop_plans[
                model_endpoints[role].manifest.artifact_sha256
            ]["evidence_ids"],
            cuda_graph_mode=desktop_plans[
                model_endpoints[role].manifest.artifact_sha256
            ].get("cuda_graph_mode", "default"),
        )
        for role in sorted(model_endpoints)
    )


def _model_transitions(
    manifest: ModelManifest,
    config: CampaignModelConfiguration,
    model_endpoint: RuntimeModelEndpointCapability,
    topology: RuntimePhysicalTopology,
    composites: tuple,
    calibration: dict[str, list],
    evidence,
):
    assert config.transition_load_metric is not None
    assert config.transition_warm_metric is not None
    results = [
        row[0]
        for arm in (
            evidence.desktop_calibration_arm,
            evidence.assisted_calibration_arm,
        )
        for row in calibration[arm]
    ]
    latency_us = math.ceil(max(
            row["switch"]["unload_ms"]
            + row["switch"][config.transition_load_metric]
            + row["switch"][config.transition_warm_metric]
            for row in results
    ) * 1000)
    energy_uj = math.ceil(latency_us * 160_000 / 1000)
    model_composites = tuple(
        row for row in composites
        if row.artifact_sha256 == manifest.artifact_sha256
    )
    return model_transition_capabilities(
        model_endpoint,
        topology,
        model_composites,
        latency_us=latency_us,
        energy_uj=energy_uj,
    )


def _register_overlay_desktop_control(
    catalog: RuntimeCapabilityCatalog,
    overlay_manifest: ModelManifest,
    overlay_config,
    topology: RuntimePhysicalTopology,
    endpoints,
) -> tuple[RuntimeCapabilityCatalog, tuple[str, ...]]:
    cpu_capability = catalog.fallback
    pool = catalog.placement_profile.memory_pools[
        topology.gpu_memory_resource_id
    ]
    device = catalog.placement_profile.devices[topology.gpu_device_id]
    capacity_bytes = min(
        pool.capacity_bytes - pool.reserved_bytes,
        device.allocation_limit_bytes,
    )
    gpu_first_layer = derived_desktop_gpu_first_layer(
        overlay_manifest,
        gpu_capacity_bytes=capacity_bytes,
        gpu_reserve_bytes=overlay_config.gpu_reserve_bytes,
    )
    require(
        gpu_first_layer < overlay_manifest.block_count,
        "overlay model has no capacity-derived GPU placement",
    )
    executor_id = (
        "physical:desktop-control:"
        + overlay_manifest.artifact_sha256[7:23]
    )
    source_profiles = tuple(
        row for row in catalog.route_shape_profiles
        if row.artifact_sha256 == overlay_manifest.artifact_sha256
        and row.route_family == "whole_model"
        and row.device_ids == (topology.gpu_device_id,)
        and row.maturity == "QUALIFIED"
    )
    require(source_profiles, "qualified overlay GPU route profile")
    evidence_ids = tuple(sorted({
        evidence_id
        for row in source_profiles
        for evidence_id in row.evidence_ids
    }))
    bound_profiles = tuple(
        replace(
            row,
            selector_id=(
                row.selector_id + ":desktop-control:"
                + overlay_manifest.artifact_sha256[7:15]
            ),
            route_family="layer_placement",
            device_ids=tuple(sorted(topology.desktop_device_ids)),
            executor_id=executor_id,
        )
        for row in source_profiles
    )
    host_to_gpu = tuple(
        row.bandwidth_bytes_per_s
        for row in catalog.placement_profile.links
        if row.source_device == topology.cpu_device_id
        and row.target_device == topology.gpu_device_id
        and row.ready
    )
    require(host_to_gpu, "overlay GPU transfer profile")
    transition_latency_us = (
        10_000_000
        + math.ceil(
            overlay_manifest.tensor_bytes
            * 4_000_000
            / max(host_to_gpu)
        )
    )
    parameters = dict(cpu_capability.adapter_parameters)
    parameters.pop("cpu_affinity", None)
    catalog = materialize_desktop_control(
        catalog,
        overlay_manifest,
        topology,
        executor_id=executor_id,
        endpoint=(
            endpoints[overlay_config.endpoint_ids["desktop"]]
        ),
        backend=overlay_config.backend_ids["desktop"],
        gpu_first_layer=gpu_first_layer,
        adapter_parameters=parameters,
        evidence_ids=evidence_ids,
        transition_latency_us=transition_latency_us,
        transition_energy_uj=max(
            1,
            transition_latency_us * 200_000 // 1000,
        ),
        route_shape_profiles=bound_profiles,
    )
    return catalog, evidence_ids


def _register_overlay_whole_phone(
    catalog: RuntimeCapabilityCatalog,
    overlay_manifest: ModelManifest,
    overlay_config,
    rig,
    topology: RuntimePhysicalTopology,
    endpoints,
    evidence_ids: tuple[str, ...],
) -> RuntimeCapabilityCatalog:
    from research_dev.scheduler.adapters.android_llama_server import ncm_control_script_sha256

    server_sha256 = rig.phone.whole_server_sha256
    return materialize_whole_model_endpoint(
        catalog,
        overlay_manifest,
        executor_id="physical:" + topology.phone_device_id,
        endpoint=endpoints[overlay_config.endpoint_ids["whole_phone"]],
        backend=overlay_config.backend_ids["whole_phone"],
        adapter_parameters={
            **({"android_control_transport": rig.phone.whole_control_transport,
                "android_control_endpoint": rig.phone.whole_ncm_adb_endpoint,
                "android_control_script_sha256": ncm_control_script_sha256()}
               if rig.phone.whole_control_transport == "adb-ncm" else {}),
            "batch_size": 1024,
            "context_size": 4096,
            "cpu_device_id": topology.cpu_device_id,
            "executable_device": rig.phone.whole_executable_device,
            "execution_adapter": "android-llama-server-v1",
            "forward_port": rig.phone.whole_forward_port,
            "gpu_device_id": topology.phone_device_id,
            "model_alias": overlay_manifest.model_id,
            "parallel": 1,
            "persistent_residency": int(
                overlay_config.phone_adapter_parameters.get(
                    "persistent_residency", 0
                )
            ),
            **({"whole_model_peak_memory_bytes": overlay_config.phone_adapter_parameters[
                "whole_model_peak_memory_bytes"
            ]} if "whole_model_peak_memory_bytes" in overlay_config.phone_adapter_parameters else {}),
            "remote_library_directory": (
                rig.phone.whole_library_directory
            ),
            "remote_model_path": overlay_config.phone_artifact_path,
            "remote_port": rig.phone.whole_remote_port,
            "remote_server_path": rig.phone.whole_server_path,
            "remote_server_sha256": server_sha256,
            "request_io_protocol": "token-ids-v1",
            "token_id_bytes": 4,
            "ubatch_size": 256,
            "whole_model_power_prior_mw": 5_000,
        },
        evidence_ids=tuple(sorted({
            *evidence_ids,
            server_sha256,
        })),
        transition_latency_us=(
            int(overlay_config.phone_adapter_parameters.get(
                "transition_latency_us", 25_000_000
            ))
        ),
        transition_energy_uj=(
            int(overlay_config.phone_adapter_parameters.get(
                "transition_energy_uj", 125_000_000
            ))
        ),
        transition_energy_maturity="SHADOW",
    )


def _register_overlay_cpu(
    catalog: RuntimeCapabilityCatalog,
    manifest: ModelManifest,
    config: CampaignModelConfiguration,
    topology: RuntimePhysicalTopology,
    endpoints,
) -> RuntimeCapabilityCatalog:
    require(bool(config.cpu_evidence_ids), "overlay CPU physical evidence is required")
    control = catalog.desktop_control_by_artifact[manifest.artifact_sha256]
    parent = catalog.composite_executor_by_id[control.executor_id]
    cpu_id = "physical:cpu-parent:" + manifest.artifact_sha256[7:23]
    cpu_endpoint = endpoints[config.endpoint_ids["cpu"]]
    parameters = dict(catalog.fallback.adapter_parameters)
    parameters.update(config.cpu_runtime_parameters)
    model = RuntimeModelEndpointCapability(
        manifest=manifest,
        desktop_executor_id=parent.executor_id,
        desktop_endpoint=parent.endpoint,
        desktop_backend=parent.backend,
        desktop_gpu_first_layer=(
            manifest.block_count - int(parent.adapter_parameters["gpu_layers"])
        ),
        adapter_parameters=parent.adapter_parameters,
        desktop_evidence_ids=parent.evidence_ids,
        phone_executor_prefix=cpu_id + ":gpu-phone",
        phone_endpoint=parent.endpoint,
        phone_backend=parent.backend,
        phone_evidence_ids=catalog.executor_by_device[topology.phone_device_id].evidence_ids,
        phone_preloaded=config.phone_residency_mode == "preloaded",
        phone_resident_limit_bytes=config.phone_resident_limit_bytes,
        ffn_max_runtime_partitions=config.ffn_max_runtime_partitions,
        phone_runtime_control_protocol="decode-boundary-v1",
        phone_adapter_parameters=config.phone_adapter_parameters,
        phone_batch_plans=config.phone_batch_plans or ("split-row",),
        qualified_phone_batch_plans=config.qualified_phone_batch_plans,
        cpu_executor_id=cpu_id,
        cpu_endpoint=cpu_endpoint,
        cpu_backend=config.backend_ids["cpu"],
        cpu_phone_executor_prefix=cpu_id + ":npu",
        cpu_phone_endpoint=cpu_endpoint,
        cpu_phone_backend=config.backend_ids.get("cpu_phone", config.backend_ids["cpu"]),
        cpu_adapter_parameters=parameters,
        cpu_evidence_ids=config.cpu_evidence_ids,
    )
    # Cold-load priors are unqualified; only transition receipts can qualify them.
    source = next(
        row for row in catalog.transitions if row.executor_id == parent.executor_id
    )
    return materialize_cpu_phone_endpoints(
        catalog, model, topology, transition_latency_us=source.fixed_latency_us,
        transition_energy_uj=source.fixed_energy_uj,
    )


def _register_overlay_model(
    catalog: RuntimeCapabilityCatalog,
    overlay_config,
    manifest_cache_path,
    rig,
    topology: RuntimePhysicalTopology,
    endpoints,
) -> tuple[RuntimeCapabilityCatalog, ModelManifest | None]:
    registered_overlay_manifest = None
    if overlay_config.host_artifact_path is not None:
        require(
            overlay_config.host_artifact_path.is_file(),
            "overlay model registration",
        )
        overlay_manifest = load_cached_gguf_manifest(
            overlay_config.model_id,
            overlay_config.host_artifact_path,
            manifest_cache_path,
        )
        registered_overlay_manifest = overlay_manifest
        catalog, evidence_ids = _register_overlay_desktop_control(
            catalog, overlay_manifest, overlay_config, topology, endpoints
        )
        if "cpu" in overlay_config.endpoint_ids:
            catalog = _register_overlay_cpu(
                catalog, overlay_manifest, overlay_config, topology, endpoints
            )
        if "whole_phone" in overlay_config.endpoint_ids:
            catalog = _register_overlay_whole_phone(
                catalog,
                overlay_manifest,
                overlay_config,
                rig,
                topology,
                endpoints,
                evidence_ids,
            )
    return catalog, registered_overlay_manifest


def main() -> int:
    args = _build_parser().parse_args()

    configuration = load_scheduler_configuration(args.campaign)
    rig = configuration.rig
    model_configs = configuration.models.assisted_models
    evidence = configuration.evidence
    topology_config = rig.topology
    endpoints = rig.endpoints
    manifests = {
        config.model_key: ModelManifest.from_json(
            load(config.checked_manifest_path)
        )
        for config in model_configs
    }
    desktop_plans = measured_desktop_plans(
        configuration.models.desktop_baseline_plans_path, manifests
    )
    calibration = _calibration_results(evidence)
    memory_bytes = {
        row.device_id: row.memory_capacity_bytes for row in rig.devices
    }
    measured = load(evidence.kernel_profile_path)
    profile = _measured_placement_profile(
        measured, memory_bytes, evidence, topology_config
    )
    power = evidence.phone_power
    profile, phone_power_profiles = _apply_phone_power(
        profile, power, topology_config
    )
    helper_evidence = {device: load_helper_evidence(path, server_path=rig.binaries["server"])
                       for device, path in evidence.helper_phone_evidence_paths.items()}
    require(all(device == row.worker.device_id for device, row in helper_evidence.items()),
            "helper evidence device mapping differs")
    if helper_evidence:
        profile = extend_helper_profile(profile, helper_evidence)
        phone_power_profiles += tuple(row.power for row in helper_evidence.values())
    phone_sessions = phone_sessions_from_json(
        load(evidence.phone_session_discovery_path)
    )
    if phone_sessions:
        profile = add_phone_session_memory_pools(profile, phone_sessions)
    kernel_evidence = "sha256:" + digest(evidence.kernel_profile_path)
    topology = _physical_topology(rig, topology_config, phone_sessions, helper_evidence)
    arm_evidence = {
        arm: tuple(sorted({
            evidence_id
            for _, _, evidence_ids in rows
            for evidence_id in evidence_ids
        }))
        for arm, rows in calibration.items()
    }
    model_config_by_key = {
        row.model_key: row for row in model_configs
    }
    speculative_rows = configuration.campaign.speculative_rows or {}
    require(set(speculative_rows) <= {row.model_id for row in model_configs},
            "speculative rows name a model that is not an assisted model")
    model_endpoints = {}
    for role, manifest in manifests.items():
        model_endpoints[role] = _model_endpoint(
            role,
            manifest,
            model_config_by_key[role],
            endpoints,
            desktop_plans,
            arm_evidence,
            evidence,
            _speculative_parameters(
                speculative_rows.get(model_config_by_key[role].model_id),
                manifest,
                model_config_by_key[role],
                desktop_plans[manifest.artifact_sha256],
            ),
        )
    withheld_co_helpers = {}
    declared_helpers = set()
    for role, manifest in manifests.items():
        declaration = helper_phone_co_helpers(rig, model_config_by_key[role], manifest)
        if declaration is not None:
            require(
                not declaration.layer_mask >> model_endpoints[role].desktop_gpu_first_layer,
                f"co-helper layers of {role} must be CPU-parent layers",
            )
            declared_helpers.update(declaration.device_ids)
            if set(declaration.device_ids) <= set(helper_evidence):
                for device in declaration.device_ids:
                    rig_row = next(row for row in rig.helper_phones if row.device_id == device)
                    validate_helper_declaration(declaration, manifest, helper_evidence[device], rig_row)
                model_endpoints[role] = replace(model_endpoints[role], co_helpers=declaration)
            else:
                withheld_co_helpers[role] = declaration.declaration_sha256
    require(set(helper_evidence) <= declared_helpers, "helper evidence has no owning model")
    composites = tuple(
        composite
        for role in sorted(model_endpoints)
        for composite in model_composite_capabilities(
            model_endpoints[role], topology
        )
    )
    desktop_controls = _desktop_controls(
        model_endpoints, composites, desktop_plans
    )
    validate_desktop_control_bindings(desktop_plans, desktop_controls)
    resources = resource_profiles_for_catalog(
        topology,
        composites,
        tuple(row.link_id for row in profile.links),
    )
    transitions = []
    for role, manifest in manifests.items():
        transitions.extend(_model_transitions(
            manifest,
            model_config_by_key[role],
            model_endpoints[role],
            topology,
            composites,
            calibration,
            evidence,
        ))
    base_executors = base_executor_capabilities(
        topology,
        evidence_id=kernel_evidence,
        phone_minimum_battery_ppm=power.minimum_battery_ppm,
        qualified_helper_phone_ids=tuple(helper_evidence),
    )
    base_executors = tuple(replace(row, evidence_ids=(
        helper_evidence[row.device_id].identity.identity_sha256,
        "sha256:" + digest(helper_evidence[row.device_id].path),
    )) if row.device_id in helper_evidence else row for row in base_executors)
    catalog = RuntimeCapabilityCatalog(
        catalog_id=rig.rig_id,
        placement_profile=profile,
        resources=resources,
        executors=base_executors,
        composite_executors=composites,
        desktop_control_profiles=desktop_controls,
        transitions=tuple(transitions),
        minimum_energy_saving_ppm=(
            configuration.campaign.minimum_energy_saving_ppm
        ),
        maximum_latency_ppm=configuration.campaign.maximum_latency_ppm,
        route_shape_profiles=route_profiles(
            model_configs,
            manifests,
            calibration,
            (
                topology.cpu_device_id,
                topology.gpu_device_id,
                topology.phone_device_id,
            ),
            evidence.desktop_calibration_arm,
            evidence.assisted_calibration_arm,
        ),
        system_cost_profiles=(),
        phone_power_profiles=phone_power_profiles,
    )
    require(
        evidence.overlay_catalog_path is not None,
        "overlay catalog is required for catalog materialization",
    )
    overlay = RuntimeCapabilityCatalog.from_json(
        load(evidence.overlay_catalog_path)
    )
    overlay = extend_helper_overlay(catalog, overlay, helper_evidence)
    catalog = merge_runtime_capability_catalogs(catalog, overlay)
    overlay_config = configuration.models.overlay_model
    catalog, registered_overlay_manifest = _register_overlay_model(
        catalog,
        overlay_config,
        configuration.models.manifest_cache_path,
        rig,
        topology,
        endpoints,
    )
    catalog = replace(
        catalog,
        minimum_energy_saving_ppm=configuration.campaign.minimum_energy_saving_ppm,
        maximum_latency_ppm=configuration.campaign.maximum_latency_ppm,
    )
    validate_final_catalog(
        catalog,
        tuple(manifests.values()),
        registered_overlay_manifest,
        require_whole_phone=("whole_phone" in overlay_config.endpoint_ids),
    )
    serialized = catalog.to_json()
    RuntimeCapabilityCatalog.from_json(serialized)
    write_new(args.output, serialized)
    print(json.dumps({
        "catalog_sha256": "sha256:" + hashlib.sha256(canonical(serialized)).hexdigest(),
        "composite_executors": [
            row.executor_id for row in catalog.composite_executors
        ],
        "output": str(args.output),
        "route_profiles": len(catalog.route_shape_profiles),
        "schema": "s42-f16-automated-catalog-materialization-v1",
        "transitions": len(catalog.transitions),
        **({"withheld_co_helpers": withheld_co_helpers} if withheld_co_helpers else {}),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
