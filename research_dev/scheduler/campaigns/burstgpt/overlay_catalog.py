#!/usr/bin/env python3
"""Convert qualified physical measurements into a runtime capability catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    PlacementHardwareProfile,
    ResourceProfile,
    RuntimeCapabilityCatalog,
    RuntimeCompositeExecutorCapability,
    RuntimeExecutorCapability,
    RuntimeRouteShapeProfile,
    RuntimeSystemCostProfile,
)
from research_dev.scheduler.adapters import (  # noqa: E402
    compile_static_column_split_contract,
)


OPERATOR_KINDS = (
    "attention",
    "attention_projection",
    "embedding",
    "ffn",
    "kv_cache",
    "lm_head",
)
ROUTE_DEVICES = {
    "desktop-cpu": "desktop-cpu",
    "desktop-cuda": "desktop-cuda",
    "phone-adreno": "op15-phone",
}
BACKEND_DEVICES = {
    "adreno": "op15-phone",
    "cpu": "desktop-cpu",
    "cuda": "desktop-cuda",
}
LINK_DEVICES = {
    "cpu": "desktop-cpu",
    "cuda": "desktop-cuda",
    "phone-memory": "op15-phone",
}
LARGE_PHASE_IDS = {
    "idle": 0,
    "qwen": 1,
    "switching": 2,
    "gemma": 3,
}
RUNTIME_AUTO_AUDIT_SCHEMA = "s42-fp16-llama1b-runtime-auto-profile-v1"
NATURAL_AUDIT_SCHEMA = "s42-fp16-llama1b-natural-route-calibration-v1"
SPLIT_ROUTE = "cpu-phone-ffn-split"
FFN_MANIFEST_SCHEMA = "s42-llama-dense-ffn-manifest-v1"
FFN_POLICY_SCHEMA = "s42-llama-ffn-vq-compiled-policy-v1"
CUDA_REQUEST_SLOTS = 8


class CatalogMaterializationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogMaterializationError(message)


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


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def verified_record(path: Path, schema: str) -> dict[str, Any]:
    value = load_object(path)
    supplied = value.get("record_sha256")
    unsigned = {
        key: row for key, row in value.items() if key != "record_sha256"
    }
    require(
        value.get("schema") == schema
        and type(supplied) is str
        and supplied == hashlib.sha256(canonical(unsigned)).hexdigest(),
        f"record identity: {path}",
    )
    return value


def physical_split_contract(
    route_profile: dict[str, Any],
    *,
    manifest_path: Path | None,
    policy_path: Path | None,
    endpoint: str | None,
    model_sha256: str,
    model_bytes: int,
) -> dict[str, Any] | None:
    routes = route_profile.get("routes")
    has_split = type(routes) is list and any(
        row.get("route_id") == SPLIT_ROUTE
        for row in routes
        if type(row) is dict
    )
    supplied = (manifest_path, policy_path, endpoint)
    if not has_split:
        require(all(value is None for value in supplied), "unused split input")
        return None
    require(all(value is not None for value in supplied), "split input set")
    assert manifest_path is not None
    assert policy_path is not None
    assert endpoint is not None
    manifest = verified_record(manifest_path, FFN_MANIFEST_SCHEMA)
    policy = verified_record(policy_path, FFN_POLICY_SCHEMA)
    binding = route_profile.get("ffn_split_binding")
    buckets = policy.get("compiled_buckets")
    model = manifest.get("model")
    require(
        type(binding) is dict
        and binding.get("route_id") == SPLIT_ROUTE
        and binding.get("route_admission")
            == "QUALIFIED_FOR_RUNTIME_SELECTION"
        and binding.get("manifest_record_sha256")
            == manifest["record_sha256"]
        and binding.get("policy_text") == policy.get("policy_text")
        and binding.get("compiled_buckets_sha256")
            == hashlib.sha256(canonical(buckets)).hexdigest(),
        "qualified split binding",
    )
    require(
        type(model) is dict
        and model.get("sha256") == model_sha256
        and model.get("size_bytes") == model_bytes
        and type(buckets) is list
        and buckets,
        "split model identity",
    )
    contract = compile_static_column_split_contract(
        manifest,
        policy,
        protocol_namespace="s41-server-ffn-static-v1",
    )
    return {
        "endpoint": endpoint,
        "evidence_ids": tuple(sorted({
            "sha256:" + digest(manifest_path),
            "sha256:" + digest(policy_path),
            "sha256:" + manifest["record_sha256"],
            "sha256:" + policy["record_sha256"],
        })),
        "fraction_ppm": contract.fraction_ppm,
        "operator_ids": contract.operator_ids,
        "protocol": contract.protocol,
    }


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(), 1):
        row = json.loads(raw)
        require(type(row) is dict, f"trace row {line_number}")
        rows.append(row)
    require(rows, "trace is empty")
    return rows


def median_integer(values: list[int]) -> int:
    require(values and all(type(value) is int for value in values), "median")
    return max(1, int(statistics.median(values)))


def placement_profile(
    measured: dict[str, Any],
    memory_bytes: dict[str, int],
) -> PlacementHardwareProfile:
    domains = []
    for row in measured["energy_domains"]:
        domains.append({
            "domain_id": row["domain_id"],
            "evidence_ids": [
                item["sha256"] for item in row.get("evidence", [])
            ],
            "idle_power_mw": row["idle_power_mw"]["median"],
            "status": "measured",
        })

    kernel_rows = measured["kernel_rows"]
    kernels = []
    for backend, device_id in BACKEND_DEVICES.items():
        matches = [row for row in kernel_rows if row["backend"] == backend]
        require(matches, f"kernel evidence absent: {backend}")
        ops_per_s = median_integer([
            row["effective_ops_per_s"] for row in matches
        ])
        active_power_mw = median_integer([
            sum(
                domain["median"]
                for domain in row["power_mw_by_domain"].values()
            )
            for row in matches
        ])
        evidence = sorted({
            item["sha256"]
            for row in matches
            for item in row.get("evidence", [])
        })
        for operator_kind in OPERATOR_KINDS:
            kernels.append({
                "active_power_mw": active_power_mw,
                "device_id": device_id,
                "domain_id": {
                    "desktop-cpu": "cpu-package",
                    "desktop-cuda": "gpu-board",
                    "op15-phone": "phone-system",
                }[device_id],
                "effective_bytes_per_s": max(1, ops_per_s // 4),
                "effective_ops_per_s": ops_per_s,
                "evidence_ids": evidence,
                "kernel_id": f"prior:{device_id}:{operator_kind}",
                "launch_us": 1,
                "profile_id": f"prior:{device_id}:{operator_kind}",
                "status": "estimated",
            })

    links = []
    for row in measured["derived_link_models"]:
        source = LINK_DEVICES[row["source_device"]]
        target = LINK_DEVICES[row["target_device"]]
        links.append({
            "bandwidth_bytes_per_s": row["bandwidth_bytes_per_s"],
            "domain_active_power_mw": {},
            "dynamic_pj_per_byte": row["dynamic_pj_per_byte"],
            "evidence_ids": row["evidence_ids"],
            "fixed_dynamic_uj": row["fixed_dynamic_uj"],
            "fixed_latency_us": row["fixed_latency_us"],
            "link_id": row["model_id"],
            "ready": True,
            "source_device": source,
            "status": "measured",
            "target_device": target,
        })

    return PlacementHardwareProfile.from_json({
        "devices": [
            {
                "allocation_limit_bytes": memory_bytes[device_id],
                "device_id": device_id,
                "kind": kind,
                "memory_pool_id": pool_id,
                "ready": True,
            }
            for device_id, kind, pool_id in (
                ("desktop-cpu", "cpu", "host-ram"),
                ("desktop-cuda", "gpu", "cuda0-vram"),
                ("op15-phone", "phone", "op15-ram"),
            )
        ],
        "domains": domains,
        "energy_boundary_id": measured["energy_boundary"]["id"],
        "idle_charge_domains": sorted(
            row["domain_id"] for row in measured["energy_domains"]
        ),
        "kernels": kernels,
        "links": links,
        "memory_pools": [
            {
                "capacity_bytes": memory_bytes[device_id],
                "pool_id": pool_id,
                "reserved_bytes": 0,
            }
            for device_id, pool_id in (
                ("desktop-cpu", "host-ram"),
                ("desktop-cuda", "cuda0-vram"),
                ("op15-phone", "op15-ram"),
            )
        ],
        "profile_id": "physical-4060ti-op15-automated-priors-v1",
        "schema": "s42-placement-hardware-profile-v1",
    })


def _runtime_auto_feature_ranges(class_id: int) -> dict[str, tuple[int, int]]:
    require(1 <= class_id <= 8, "runtime contention class")
    phase_class = ((class_id - 1) % 4) + 1
    phase_id = 0 if phase_class == 4 else phase_class
    phone_assistance = int(class_id > 4)
    return {
        "large_model_op15": (phone_assistance, phone_assistance),
        "large_phase_id": (phase_id, phase_id),
    }


def latency_variants(
    route: dict[str, Any], *, runtime_auto: bool
) -> list[dict[str, Any]]:
    latency = route["latency"]
    if latency.get("kind") == "conditioned_affine_features_v1":
        selector = latency["selector_feature"]
        return [
            {
                "feature_ranges": {
                    **(
                        _runtime_auto_feature_ranges(row["selector_value"])
                        if runtime_auto
                        and selector == "contention_class_id"
                        else {
                            selector: (
                                row["selector_value"], row["selector_value"]
                            )
                        }
                    )
                },
                "label": row["label"],
                "model": row["cost_us"],
                "sample_count": row["sample_count"],
                "ucb_add_us": row["ucb_add_us"],
            }
            for row in latency["variants"]
        ]
    return [{
        "feature_ranges": {},
        "label": route["route_id"],
        "model": latency["cost_us"],
        "sample_count": latency["sample_count"],
        "ucb_add_us": latency["ucb_add_us"],
    }]


def route_profiles(
    route_profile: dict[str, Any],
    audit: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    artifact_sha256: str,
    evidence_sha256: str,
    split_contract: dict[str, Any] | None,
) -> tuple[RuntimeRouteShapeProfile, ...]:
    runtime_auto = audit.get("schema") == RUNTIME_AUTO_AUDIT_SCHEMA
    fitted = (
        {row["route_id"] for row in route_profile["routes"]}
        if runtime_auto
        else set(audit["fitted_routes"])
    )
    input_min = min(row["input_tokens"] for row in trace_rows)
    input_max = max(row["input_tokens"] for row in trace_rows)
    output_min = min(row["output_tokens"] for row in trace_rows)
    output_max = max(row["output_tokens"] for row in trace_rows)
    result = []
    for route in route_profile["routes"]:
        route_id = route["route_id"]
        if route_id not in fitted:
            continue
        energy = route["energy"]
        require(
            energy.get("status") == "measured"
            and route.get("placement_verified") is True
            and route.get("resident") is True,
            f"qualified route evidence differs: {route_id}",
        )
        energy_cost = energy["cost_uj"]
        require(
            energy_cost.get("kind") == "affine_tokens_v1",
            f"route energy model differs: {route_id}",
        )
        if route_id in ROUTE_DEVICES:
            route_family = "whole_model"
            device_ids = (ROUTE_DEVICES[route_id],)
            assisted_operator_kind = None
            split_axis = "none"
            split_fraction_ppm = 0
        else:
            require(
                route_id == SPLIT_ROUTE and split_contract is not None,
                f"physical route contract absent: {route_id}",
            )
            route_family = "operator_split"
            device_ids = ("desktop-cpu", "op15-phone")
            assisted_operator_kind = "ffn"
            split_axis = "column"
            split_fraction_ppm = split_contract["fraction_ppm"]
        for index, variant in enumerate(
            latency_variants(route, runtime_auto=runtime_auto)
        ):
            model = variant["model"]
            require(
                model.get("kind") in {
                    "affine_features_v1", "affine_tokens_v1"
                },
                f"route latency model differs: {route_id}",
            )
            coefficients = dict(model.get("coefficients", {}))
            input_coefficient = model.get(
                "input_token", coefficients.pop("input_tokens", 0)
            )
            output_coefficient = model.get(
                "output_token", coefficients.pop("output_tokens", 0)
            )
            feature_ranges = dict(variant["feature_ranges"])
            for name in coefficients:
                feature_ranges.setdefault(name, (0, 2**31 - 1))
            result.append(RuntimeRouteShapeProfile(
                selector_id=(
                    f"physical:{route_id}:{variant['label']}:{index}"
                ),
                artifact_sha256="sha256:" + artifact_sha256,
                route_family=route_family,
                device_ids=device_ids,
                assisted_operator_kind=assisted_operator_kind,
                split_axis=split_axis,
                split_fraction_ppm=split_fraction_ppm,
                residency_variant="hot",
                minimum_input_tokens=input_min,
                maximum_input_tokens=input_max,
                minimum_output_tokens=output_min,
                maximum_output_tokens=output_max,
                service_fixed_us=model["fixed"],
                service_input_token_us=input_coefficient,
                service_output_token_us=output_coefficient,
                service_upper_add_us=variant["ucb_add_us"],
                energy_fixed_uj=energy_cost["fixed"],
                energy_input_token_uj=energy_cost["input_token"],
                energy_output_token_uj=energy_cost["output_token"],
                energy_lower_error_ppm=energy["lower_error_ppm"],
                energy_upper_error_ppm=energy["upper_error_ppm"],
                sample_count=variant["sample_count"],
                maturity="QUALIFIED",
                evidence_ids=tuple(sorted({
                    "sha256:" + evidence_sha256,
                    "sha256:" + audit["profile_sha256"],
                })),
                feature_ranges=feature_ranges,
                service_feature_coefficients_us=coefficients,
                energy_feature_coefficients_uj={},
            ))
    require(result, "no held-out route profile was materialized")
    return tuple(result)


def control_overhead_profile(
    path: Path | None,
) -> tuple[int, int, int, bool, tuple[str, ...]]:
    if path is None:
        return 0, 0, 1, False, ()
    value = load_object(path)
    require(
        value.get("schema")
            == "s42-automated-control-overhead-profile-v1"
        and value.get("measured") is True
        and type(value.get("mean_us")) is int
        and type(value.get("upper_us")) is int
        and type(value.get("sample_count")) is int
        and value["sample_count"] >= 2
        and 0 <= value["mean_us"] <= value["upper_us"],
        "automated control overhead profile",
    )
    return (
        value["mean_us"],
        value["upper_us"],
        value["sample_count"],
        True,
        ("sha256:" + digest(path),),
    )


def system_cost_profiles(
    profile: dict[str, Any],
    *,
    large_model_policy: str,
    resource_ids: tuple[str, ...],
    profile_sha256: str,
    control_profile_path: Path | None,
) -> tuple[RuntimeSystemCostProfile, ...]:
    require(
        profile.get("schema")
            == "s42-fp16-overlay-marginal-system-profile-v1",
        "marginal system profile schema",
    )
    arms = profile.get("arms")
    require(type(arms) is dict, "marginal system arms")
    arm = arms.get(large_model_policy)
    require(type(arm) is dict, "marginal system arm")
    phases = arm.get("phases")
    require(type(phases) is dict, "marginal system phases")
    mean_us, upper_us, control_samples, control_measured, control_evidence = (
        control_overhead_profile(control_profile_path)
    )
    evidence = tuple(sorted({
        "sha256:" + profile_sha256,
        *control_evidence,
    }))
    result = []
    for phase_name, phase_id in sorted(
        LARGE_PHASE_IDS.items(), key=lambda row: row[1]
    ):
        phase = phases.get(phase_name)
        require(type(phase) is dict, "marginal system phase")
        route_interference = phase.get("route_cpu_interference_ppm")
        require(
            type(route_interference) is dict
            and all(
                type(route_interference.get(route_id)) is int
                and route_interference[route_id] >= 0
                for route_id in ROUTE_DEVICES
            ),
            "marginal system route interference",
        )
        interference = {resource_id: 0 for resource_id in resource_ids}
        interference["desktop-cpu"] = route_interference["desktop-cpu"]
        interference["cuda0"] = route_interference["desktop-cuda"]
        interference["op15-adreno"] = route_interference["phone-adreno"]
        measured = (
            arm.get("measured") is True
            and phase.get("measured", True) is True
            and control_measured
        )
        result.append(RuntimeSystemCostProfile(
            selector_id=(
                f"physical-system:{large_model_policy}:{phase_name}"
            ),
            feature_ranges={
                "large_model_op15": (
                    int(large_model_policy == "op15-assistance"),
                    int(large_model_policy == "op15-assistance"),
                ),
                "large_phase_id": (phase_id, phase_id),
            },
            interference_ppm_by_resource=interference,
            control_delay_us=mean_us,
            control_delay_upper_us=upper_us,
            lower_error_ppm=int(arm["lower_error_ppm"]),
            upper_error_ppm=int(arm["upper_error_ppm"]),
            sample_count=min(int(arm["sample_count"]), control_samples),
            maturity="QUALIFIED" if measured else "SHADOW",
            evidence_ids=evidence,
        ))
    return tuple(result)


def executor(
    *,
    executor_id: str,
    device_id: str,
    endpoint: str,
    backend: str,
    execution_resources: tuple[str, ...],
    memory_resource_id: str,
    fallback: bool,
    evidence_id: str,
    adapter_parameters: dict[str, int | str] | None = None,
) -> RuntimeExecutorCapability:
    return RuntimeExecutorCapability(
        executor_id=executor_id,
        device_id=device_id,
        endpoint=endpoint,
        backend=backend,
        execution_resource_ids=execution_resources,
        memory_resource_id=memory_resource_id,
        kernel_profiles={
            kind: f"prior:{device_id}:{kind}" for kind in OPERATOR_KINDS
        },
        supported_quantizations=("*",),
        supports_whole_model=True,
        supports_layer_placement=True,
        supports_operator_placement=True,
        supports_kv_cache=True,
        supports_split_coordinator=True,
        supports_split_helper=True,
        split_axes=("column", "row", "tensor"),
        split_fractions_ppm=(250_000, 500_000, 750_000),
        layer_fractions_ppm=(250_000, 500_000, 750_000),
        residency_states=("hot", "warm", "cold"),
        maturity="QUALIFIED",
        evidence_ids=(evidence_id,),
        qualified_fallback=fallback,
        maximum_temperature_millic=90_000,
        minimum_battery_ppm=(200_000 if device_id == "op15-phone" else 0),
        workspace_bytes_per_token=4096,
        coordinated_route_families=(),
        operator_plan_protocol=None,
        exclusive_residency_resource_id=(
            "cuda0" if device_id == "desktop-cuda" else None
        ),
        adapter_parameters=(
            {} if adapter_parameters is None else adapter_parameters
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-profile", type=Path, required=True)
    parser.add_argument("--route-profile-audit", type=Path, required=True)
    parser.add_argument("--model-identity-audit", type=Path)
    parser.add_argument("--kernel-profile", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-bytes", type=int, required=True)
    parser.add_argument("--cpu-endpoint", required=True)
    parser.add_argument("--gpu-endpoint", required=True)
    parser.add_argument("--phone-endpoint", required=True)
    parser.add_argument("--cpu-affinity", default="20-23")
    parser.add_argument("--split-endpoint")
    parser.add_argument("--ffn-manifest", type=Path)
    parser.add_argument("--ffn-policy", type=Path)
    parser.add_argument("--host-memory-bytes", type=int, required=True)
    parser.add_argument("--gpu-memory-bytes", type=int, required=True)
    parser.add_argument("--phone-memory-bytes", type=int, required=True)
    parser.add_argument("--marginal-system-profile", type=Path, required=True)
    parser.add_argument("--large-model-policy", required=True)
    parser.add_argument("--control-overhead-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _require_qualified_evidence(
    args: argparse.Namespace,
    route_profile: dict[str, Any],
    audit: dict[str, Any],
    measured: dict[str, Any],
    profile_sha256: str,
) -> None:
    runtime_auto = audit.get("schema") == RUNTIME_AUTO_AUDIT_SCHEMA
    identity_audit = (
        load_object(args.model_identity_audit)
        if args.model_identity_audit is not None
        else audit
    )
    audit_identity_valid = (
        runtime_auto
        and args.model_identity_audit is not None
        and audit.get("input_sha256", {}).get("natural_audit")
            == digest(args.model_identity_audit)
        and identity_audit.get("schema") == NATURAL_AUDIT_SCHEMA
        and identity_audit.get("status") == "PASS"
    ) or (
        not runtime_auto
        and args.model_identity_audit is None
        and audit.get("all_variants_measured") is True
        and audit.get("input_identity", {}).get("overlay_trace")
            == digest(args.requests)
    )
    require(
        route_profile.get("schema") == "s42-general-scheduler-profile-v1"
        and audit.get("status") == "PASS"
        and audit.get("profile_sha256") == profile_sha256
        and audit_identity_valid
        and (
            not runtime_auto
            or audit.get("supported_contention_classes")
                == list(range(1, 9))
        )
        and measured.get("schema") == "s42-kernel-energy-profile-v1"
        and measured.get("qualification", {}).get("enforcement")
            == "fail_closed",
        "qualified input evidence",
    )
    model_identity = identity_audit["input_identity"]["model_identities"].get(
        "llama-3.2-1b-instruct-q4_0"
    )
    require(
        type(model_identity) is dict
        and model_identity.get("artifact_sha256") == args.model_sha256
        and model_identity.get("artifact_bytes") == args.model_bytes,
        "model identity differs from route evidence",
    )


def _resource_rows(profile: PlacementHardwareProfile) -> tuple[ResourceProfile, ...]:
    return (
        ResourceProfile("desktop-cpu", "compute", 4, True, "desktop-cpu"),
        ResourceProfile(
            "cuda0", "compute", CUDA_REQUEST_SLOTS, True, "cuda0"
        ),
        ResourceProfile("op15-adreno", "compute", 1, True, "op15-adreno"),
        ResourceProfile("op15-htp", "compute", 1, True, "op15-htp"),
        ResourceProfile("desktop-usb-root", "transport", 1, True, "usb-root"),
        ResourceProfile(
            "op15-functionfs", "transport", 1, True, "op15-functionfs"
        ),
        ResourceProfile("op15-ncm", "transport", 1, True, "op15-ncm"),
        *(
            ResourceProfile(
                "link:" + row.link_id,
                "transport",
                (
                    CUDA_REQUEST_SLOTS
                    if {row.source_device, row.target_device}
                        == {"desktop-cpu", "desktop-cuda"}
                    else 1
                ),
                True,
                row.link_id,
            )
            for row in profile.links
        ),
    )


def _composite_executors(
    split_contract: dict[str, Any] | None,
    profile: PlacementHardwareProfile,
    model_sha256: str,
) -> tuple[RuntimeCompositeExecutorCapability, ...]:
    phone_link_resources = tuple(sorted(
        "link:" + row.link_id
        for row in profile.links
        if {row.source_device, row.target_device}
            == {"desktop-cpu", "op15-phone"}
    ))
    if split_contract is None:
        return ()
    return (RuntimeCompositeExecutorCapability(
        executor_id="physical:cpu-phone-ffn-split",
        endpoint=split_contract["endpoint"],
        backend="cpu-op15-htp-ncm",
        coordinator_device_id="desktop-cpu",
        participant_device_ids=("desktop-cpu", "op15-phone"),
        participant_resource_ids={
            "desktop-cpu": ("desktop-cpu",),
            "op15-phone": (
                "desktop-usb-root",
                "op15-htp",
                "op15-ncm",
                *phone_link_resources,
            ),
        },
        route_family="operator_split",
        assisted_operator_kind="ffn",
        split_axis="column",
        split_fractions_ppm=(split_contract["fraction_ppm"],),
        layer_fractions_ppm=(),
        residency_states=("hot",),
        resource_ids=(
            "desktop-cpu",
            "desktop-usb-root",
            "op15-htp",
            "op15-ncm",
            *phone_link_resources,
        ),
        operator_plan_protocol=split_contract["protocol"],
        maturity="QUALIFIED",
        evidence_ids=split_contract["evidence_ids"],
        artifact_sha256="sha256:" + model_sha256,
        operator_ids=split_contract["operator_ids"],
    ),)


def _base_executors(
    args: argparse.Namespace, profile_sha256: str
) -> tuple[RuntimeExecutorCapability, ...]:
    return (
        executor(
            executor_id="physical:desktop-cpu",
            device_id="desktop-cpu",
            endpoint=args.cpu_endpoint,
            backend="cpu",
            execution_resources=("desktop-cpu",),
            memory_resource_id="host-ram",
            fallback=True,
            evidence_id="sha256:" + profile_sha256,
            adapter_parameters={
                "batch_size": 2048,
                "context_size": 8192,
                "cpu_affinity": args.cpu_affinity,
                "cpu_device_id": "desktop-cpu",
                "gpu_device_id": "desktop-cuda",
                "gpu_layers": 0,
                "model_alias": "llama-3.2-1b-instruct-q4_0",
                "parallel": 4,
                "threads": 4,
                "threads_batch": 4,
                "ubatch_size": 512,
            },
        ),
        executor(
            executor_id="physical:desktop-cuda",
            device_id="desktop-cuda",
            endpoint=args.gpu_endpoint,
            backend="cuda",
            execution_resources=("cuda0",),
            memory_resource_id="cuda0-vram",
            fallback=False,
            evidence_id="sha256:" + profile_sha256,
        ),
        executor(
            executor_id="physical:op15-phone",
            device_id="op15-phone",
            endpoint=args.phone_endpoint,
            backend="phone-adreno-ncm",
            execution_resources=(
                "op15-adreno", "op15-ncm", "desktop-usb-root"
            ),
            memory_resource_id="op15-ram",
            fallback=False,
            evidence_id="sha256:" + profile_sha256,
        ),
    )


def _write_catalog(output: Path, catalog: RuntimeCapabilityCatalog) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    require(not temporary.exists(), "new temporary output")
    try:
        temporary.write_bytes(canonical(catalog.to_json()))
        RuntimeCapabilityCatalog.from_json(
            json.loads(temporary.read_text(encoding="ascii"))
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _print_summary(catalog: RuntimeCapabilityCatalog) -> None:
    print(json.dumps({
        "catalog_id": catalog.catalog_id,
        "candidate_devices": [
            row.device_id for row in catalog.executors
        ],
        "composite_executors": [
            row.executor_id for row in catalog.composite_executors
        ],
        "qualified_route_profiles": [
            row.selector_id for row in catalog.route_shape_profiles
        ],
        "system_cost_profiles": [
            {
                "maturity": row.maturity,
                "selector_id": row.selector_id,
            }
            for row in catalog.system_cost_profiles
        ],
        "schema": "s42-automated-runtime-catalog-materialization-v1",
    }, sort_keys=True))


def main() -> int:
    args = _build_parser().parse_args()

    require(
        len(args.model_sha256) == 64
        and all(value in "0123456789abcdef" for value in args.model_sha256),
        "model SHA-256",
    )
    require(args.model_bytes > 0, "model bytes")
    require(args.output.is_absolute() and not args.output.exists(), "new output")
    route_profile = load_object(args.route_profile)
    audit = load_object(args.route_profile_audit)
    measured = load_object(args.kernel_profile)
    marginal = load_object(args.marginal_system_profile)
    trace_rows = load_trace(args.requests)
    split_contract = physical_split_contract(
        route_profile,
        manifest_path=args.ffn_manifest,
        policy_path=args.ffn_policy,
        endpoint=args.split_endpoint,
        model_sha256=args.model_sha256,
        model_bytes=args.model_bytes,
    )
    profile_sha256 = digest(args.route_profile)
    _require_qualified_evidence(
        args, route_profile, audit, measured, profile_sha256
    )
    memory = {
        "desktop-cpu": args.host_memory_bytes,
        "desktop-cuda": args.gpu_memory_bytes,
        "op15-phone": args.phone_memory_bytes,
    }
    require(all(value > 0 for value in memory.values()), "memory capacity")
    profile = placement_profile(measured, memory)
    resource_rows = _resource_rows(profile)
    resource_ids = tuple(row.resource_id for row in resource_rows)
    composite_executors = _composite_executors(
        split_contract, profile, args.model_sha256
    )
    catalog = RuntimeCapabilityCatalog(
        catalog_id="physical-4060ti-op15-automated-runtime-v1",
        placement_profile=profile,
        resources={row.resource_id: row for row in resource_rows},
        executors=_base_executors(args, profile_sha256),
        composite_executors=composite_executors,
        transitions=(),
        minimum_energy_saving_ppm=route_profile["policy"][
            "energy_saving_ppm"
        ],
        maximum_latency_ppm=1_000_000,
        route_shape_profiles=route_profiles(
            route_profile,
            audit,
            trace_rows,
            args.model_sha256,
            profile_sha256,
            split_contract,
        ),
        system_cost_profiles=tuple(
            profile
            for large_model_policy in (
                ("cpu-overflow", "op15-assistance")
                if args.large_model_policy == "all"
                else (args.large_model_policy,)
            )
            for profile in system_cost_profiles(
                marginal,
                large_model_policy=large_model_policy,
                resource_ids=resource_ids,
                profile_sha256=digest(args.marginal_system_profile),
                control_profile_path=args.control_overhead_profile,
            )
        ),
    )
    _write_catalog(args.output, catalog)
    _print_summary(catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
