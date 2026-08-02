#!/usr/bin/env python3
"""Aggregate a complete, prospectively ordered S40 primary campaign."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import tempfile
from typing import Any

from campaign_plan import PRIMARY_ROTATIONS, read_campaign
from evidence_common import (
    EvidenceError,
    canonical_bytes,
    digest_bytes,
    digest_file,
    parse_json,
    percentile,
    read_json,
    require,
    require_int,
    require_string,
    validate_digest,
)
from event_evidence import reduce_paths
from run_manifest import validate_run_manifest
from validate_inputs import DEFAULT_CONTRACT


RUN_KEYS = {
    "cache_regime",
    "manifest_sha256",
    "metrics",
    "mode",
    "order",
    "performance_claim_authorized",
    "phase",
    "repeat_index",
    "run_id",
}
METRIC_KEYS = {
    "cleanup_count",
    "cleanup_p50_ns",
    "cleanup_p95_ns",
    "completed_request_count",
    "completion_latency_p50_ns",
    "completion_latency_p95_ns",
    "completion_latency_p99_ns",
    "controller_process_cpu_utilization_milli_pct_p50",
    "controller_process_swap_growth_bytes",
    "cpu_utilization_milli_pct_p50",
    "discard_count",
    "discard_p50_ns",
    "discard_p95_ns",
    "drain_count",
    "drain_p50_ns",
    "drain_p95_ns",
    "gpu_energy_scope",
    "load_count",
    "load_p50_ns",
    "load_p95_ns",
    "maximum_global_token_publication_gap_ns",
    "maximum_model_publication_gap_ns",
    "minimum_system_mem_available_bytes",
    "op12_all_interface_rx_bytes",
    "op12_all_interface_tx_bytes",
    "op12_minimum_available_bytes",
    "op12_swap_growth_bytes",
    "op12_thermal_max_millic",
    "op15_all_interface_rx_bytes",
    "op15_all_interface_tx_bytes",
    "op15_minimum_available_bytes",
    "op15_swap_growth_bytes",
    "op15_thermal_max_millic",
    "ownership_commit_count",
    "ownership_commit_latency_p95_ns",
    "peak_gpu_memory_used_bytes",
    "peak_controller_process_rss_bytes",
    "queue_p50_ns",
    "queue_p95_ns",
    "queue_p99_ns",
    "replay_count",
    "replay_p50_ns",
    "replay_p95_ns",
    "selected_gpu_board_energy_nj",
    "slo_goodput_milli_rps",
    "slo_met_count",
    "stranded_request_count",
    "system_swap_growth_bytes",
    "tokens_per_second_milli",
    "ttft_p50_ns",
    "ttft_p95_ns",
    "ttft_p99_ns",
    "unload_count",
    "unload_p50_ns",
    "unload_p95_ns",
}
INTEGER_METRICS = METRIC_KEYS - {"gpu_energy_scope"}
OPTIONAL_LATENCY_METRICS = {
    "completion_latency_p50_ns",
    "completion_latency_p95_ns",
    "completion_latency_p99_ns",
    "queue_p50_ns",
    "queue_p95_ns",
    "queue_p99_ns",
    "ttft_p50_ns",
    "ttft_p95_ns",
    "ttft_p99_ns",
}
PHASES = ("cleanup", "discard", "drain", "load", "replay", "unload")
PHASE_LATENCY_METRICS = {
    f"{phase}_{percentile_name}_ns"
    for phase in PHASES
    for percentile_name in ("p50", "p95")
}
PHONE_METRICS = {
    f"{phone}_{metric}"
    for phone in ("op12", "op15")
    for metric in (
        "all_interface_rx_bytes",
        "all_interface_tx_bytes",
        "minimum_available_bytes",
        "swap_growth_bytes",
        "thermal_max_millic",
    )
}
OPTIONAL_METRICS = (
    OPTIONAL_LATENCY_METRICS
    | PHASE_LATENCY_METRICS
    | PHONE_METRICS
    | {"ownership_commit_latency_p95_ns"}
)
REQUIRED_INTEGER_METRICS = INTEGER_METRICS - OPTIONAL_METRICS
PRIMARY_CELLS = (
    ("C1_GPU_ONLY_OPTIMIZED", "WARM_HOST_CACHE"),
    ("C1_GPU_ONLY_OPTIMIZED", "COLD_NVME"),
    ("C2_GPU_PLUS_CPU_WARM_EXECUTOR", "WARM_HOST_CACHE"),
    ("T1_PHONE_WARM_TIER", "WARM_HOST_CACHE"),
)
PHYSICAL_ARTIFACT_ROLES = {
    "controller_events",
    "controller_launch",
    "resource_samples",
    "trace_start",
}


def expected_primary(t2_repetitions: int) -> list[dict[str, Any]]:
    require(t2_repetitions in {1, 3},
            "campaign reduction: T2 repetitions must be 1 or 3")
    result = []
    order = 0
    for repeat_index, rotation in enumerate(PRIMARY_ROTATIONS):
        for mode, cache_regime in rotation:
            result.append({
                "cache_regime": cache_regime,
                "mode": mode,
                "order": order,
                "phase": "PRIMARY",
                "repeat_index": repeat_index,
            })
            order += 1
    for repeat_index in range(t2_repetitions):
        result.append({
            "cache_regime": "WARM_HOST_CACHE",
            "mode": "T2_PHONE_NO_PROMOTION",
            "order": order,
            "phase": "T2_ISOLATION",
            "repeat_index": repeat_index,
        })
        order += 1
    return result


def _contained_path(root: Path, value: Any, field: str) -> Path:
    relative = Path(require_string(value, field))
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"{field}: expected contained relative path",
    )
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise EvidenceError(
            f"{field}: path escapes run directory") from error
    require(path.is_file(), f"{field}: missing file")
    return path


def _snapshot_artifacts(
        manifest_path: Path,
        manifest: dict[str, Any],
) -> dict[str, bytes]:
    root = manifest_path.parent
    records = manifest.get("artifacts")
    require(isinstance(records, list), "physical campaign: missing artifacts")
    snapshots: dict[str, bytes] = {}
    for index, record in enumerate(records):
        field = f"physical campaign artifact[{index}]"
        require(isinstance(record, dict), f"{field}: expected object")
        role = record.get("role")
        if role not in PHYSICAL_ARTIFACT_ROLES:
            continue
        require(role not in snapshots, f"{field}: duplicate role")
        path = _contained_path(root, record.get("path"), f"{field}.path")
        raw = path.read_bytes()
        require(
            len(raw) == require_int(record.get("bytes"), f"{field}.bytes")
            and digest_bytes(raw)
            == validate_digest(record.get("sha256"), f"{field}.sha256"),
            f"{field}: byte binding mismatch",
        )
        if record.get("format") == "JSONL":
            require(
                raw.endswith(b"\n")
                and len(raw.splitlines())
                == require_int(
                    record.get("record_count"),
                    f"{field}.record_count",
                    1,
                ),
                f"{field}: JSONL record count mismatch",
            )
        snapshots[role] = raw
    require(
        set(snapshots) == PHYSICAL_ARTIFACT_ROLES,
        "physical campaign: required raw artifact is missing",
    )
    return snapshots


def _reduce_snapshots(
        snapshots: dict[str, bytes],
        requests_raw: bytes,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
            prefix="s40_campaign_reduce_") as directory:
        root = Path(directory)
        paths = {}
        for role, raw in snapshots.items():
            suffix = ".jsonl" if role in {
                "controller_events", "resource_samples"} else ".json"
            path = root / f"{role}{suffix}"
            path.write_bytes(raw)
            paths[role] = path
        requests_path = root / "requests.jsonl"
        requests_path.write_bytes(requests_raw)
        return reduce_paths(
            paths["controller_events"],
            requests_path,
            paths["resource_samples"],
            paths["trace_start"],
        )


def _manifest_artifact_digest(
        manifest: dict[str, Any],
        role: str,
) -> str:
    matches = [
        row for row in manifest["artifacts"] if row.get("role") == role
    ]
    require(
        len(matches) == 1,
        f"physical campaign: missing artifact role {role}",
    )
    return validate_digest(
        matches[0].get("sha256"),
        f"physical campaign artifact {role}.sha256",
    )


def _validate_metrics(
        value: Any,
        field: str,
        phone_required: bool | None = None) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == METRIC_KEYS,
            f"{field}: metric key set mismatch")
    for name in sorted(REQUIRED_INTEGER_METRICS):
        require_int(value[name], f"{field}.{name}")
    completed = require_int(
        value["completed_request_count"],
        f"{field}.completed_request_count",
    )
    for name in sorted(OPTIONAL_LATENCY_METRICS):
        if completed == 0:
            require(
                value[name] is None,
                f"{field}.{name}: expected null with no completed requests",
            )
        else:
            require_int(value[name], f"{field}.{name}")
    for phase in PHASES:
        count = require_int(
            value[f"{phase}_count"], f"{field}.{phase}_count")
        for percentile_name in ("p50", "p95"):
            name = f"{phase}_{percentile_name}_ns"
            if count == 0:
                require(
                    value[name] is None,
                    f"{field}.{name}: expected null with no phase samples",
                )
            else:
                require_int(value[name], f"{field}.{name}")
    commit_count = require_int(
        value["ownership_commit_count"],
        f"{field}.ownership_commit_count",
    )
    if commit_count == 0:
        require(
            value["ownership_commit_latency_p95_ns"] is None,
            f"{field}.ownership_commit_latency_p95_ns: "
            "expected null with no ownership commits",
        )
    else:
        require_int(
            value["ownership_commit_latency_p95_ns"],
            f"{field}.ownership_commit_latency_p95_ns",
        )
    phone_values = [value[name] for name in sorted(PHONE_METRICS)]
    require(
        all(item is None for item in phone_values)
        or all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and item >= 0
            for item in phone_values
        ),
        f"{field}: phone metrics must be all-null or all-integer",
    )
    if phone_required is not None:
        require(
            (phone_required and all(item is not None for item in phone_values))
            or (
                not phone_required
                and all(item is None for item in phone_values)
            ),
            f"{field}: phone metric applicability mismatch",
        )
    require(
        value["gpu_energy_scope"]
        == "SELECTED_GPU_BOARD_DEVELOPMENT_ONLY",
        f"{field}: invalid GPU energy scope",
    )
    require(
        value["completed_request_count"]
        + value["stranded_request_count"] == 74,
        f"{field}: request conservation mismatch",
    )
    require(
        value["slo_met_count"] <= value["completed_request_count"],
        f"{field}: SLO count exceeds completions",
    )
    return value


def _median(values: list[int]) -> int:
    return percentile(values, 1, 2)


def _conservative_median(values: list[int | None]) -> int | None:
    if any(value is None for value in values):
        return None
    return _median(values)


def _ratio_milli(numerator: int, denominator: int) -> int | None:
    return (
        numerator * 1000 // denominator
        if denominator > 0 else None
    )


def _phone_metrics(
        phone_summary: dict[str, Any] | None) -> dict[str, int | None]:
    if phone_summary is None:
        return {name: None for name in PHONE_METRICS}
    require(
        isinstance(phone_summary, dict)
        and phone_summary.get("status") == "PASS",
        "phone summary: invalid validated summary",
    )
    result: dict[str, int | None] = {}
    for phone in ("op12", "op15"):
        interfaces = phone_summary["interface_byte_deltas"][phone]
        require(
            isinstance(interfaces, dict) and interfaces,
            f"phone summary: missing {phone} interface counters",
        )
        result[f"{phone}_all_interface_rx_bytes"] = sum(
            require_int(
                row["rx_bytes"],
                f"phone summary.{phone}.{name}.rx_bytes",
            )
            for name, row in interfaces.items()
        )
        result[f"{phone}_all_interface_tx_bytes"] = sum(
            require_int(
                row["tx_bytes"],
                f"phone summary.{phone}.{name}.tx_bytes",
            )
            for name, row in interfaces.items()
        )
        result[f"{phone}_minimum_available_bytes"] = require_int(
            phone_summary["memory_min_bytes"][phone],
            f"phone summary.{phone}.memory_min_bytes",
        )
        result[f"{phone}_swap_growth_bytes"] = require_int(
            phone_summary["swap_growth_bytes"][phone],
            f"phone summary.{phone}.swap_growth_bytes",
        )
        result[f"{phone}_thermal_max_millic"] = require_int(
            phone_summary["thermal_max_millic"][phone],
            f"phone summary.{phone}.thermal_max_millic",
        )
    return result


def metrics_from_reduced(
        value: Any,
        phone_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    require(isinstance(value, dict), "reduced run: expected object")
    require(value.get("verdict") == "PASS", "reduced run: fail-closed verdict")
    energy = value.get("energy")
    require(isinstance(energy, dict), "reduced run: missing energy")
    require(
        energy.get("gpu_energy_scope")
        == "SELECTED_GPU_BOARD_DEVELOPMENT_ONLY"
        and energy.get("energy_claim_authorized") is False,
        "reduced run: invalid selected-GPU energy scope",
    )
    phases = value.get("phase_duration_ns")
    require(
        isinstance(phases, dict) and set(phases) == set(PHASES),
        "reduced run: invalid phase durations",
    )
    phase_metrics = {}
    for phase in PHASES:
        row = phases[phase]
        require(
            isinstance(row, dict) and set(row) == {"count", "p50", "p95"},
            f"reduced run: invalid {phase} phase row",
        )
        phase_metrics.update({
            f"{phase}_count": row["count"],
            f"{phase}_p50_ns": row["p50"],
            f"{phase}_p95_ns": row["p95"],
        })
    result = {
        **phase_metrics,
        "completed_request_count": value.get("completed_request_count"),
        "completion_latency_p50_ns":
            value.get("completion_latency_p50_ns"),
        "completion_latency_p95_ns":
            value.get("completion_latency_p95_ns"),
        "completion_latency_p99_ns":
            value.get("completion_latency_p99_ns"),
        "gpu_energy_scope": energy.get("gpu_energy_scope"),
        "controller_process_cpu_utilization_milli_pct_p50":
            energy.get("controller_process_cpu_utilization_milli_pct_p50"),
        "controller_process_swap_growth_bytes":
            energy.get("controller_process_swap_growth_bytes"),
        "cpu_utilization_milli_pct_p50":
            energy.get("cpu_utilization_milli_pct_p50"),
        "maximum_global_token_publication_gap_ns":
            value.get("maximum_global_token_publication_gap_ns"),
        "maximum_model_publication_gap_ns":
            value.get("maximum_model_publication_gap_ns"),
        "minimum_system_mem_available_bytes":
            energy.get("minimum_system_mem_available_bytes"),
        **_phone_metrics(phone_summary),
        "ownership_commit_count":
            value.get("ownership_commit_count"),
        "ownership_commit_latency_p95_ns":
            value.get("ownership_commit_latency_p95_ns"),
        "peak_gpu_memory_used_bytes":
            energy.get("peak_gpu_memory_used_bytes"),
        "peak_controller_process_rss_bytes":
            energy.get("peak_controller_process_rss_bytes"),
        "queue_p50_ns": value.get("queue_p50_ns"),
        "queue_p95_ns": value.get("queue_p95_ns"),
        "queue_p99_ns": value.get("queue_p99_ns"),
        "selected_gpu_board_energy_nj": energy.get("gpu_energy_nj"),
        "slo_goodput_milli_rps": value.get("slo_goodput_milli_rps"),
        "slo_met_count": value.get("slo_met_count"),
        "stranded_request_count": value.get("stranded_request_count"),
        "system_swap_growth_bytes": energy.get("system_swap_growth_bytes"),
        "tokens_per_second_milli": value.get("tokens_per_second_milli"),
        "ttft_p50_ns": value.get("ttft_p50_ns"),
        "ttft_p95_ns": value.get("ttft_p95_ns"),
        "ttft_p99_ns": value.get("ttft_p99_ns"),
    }
    return _validate_metrics(
        result,
        "reduced run.metrics",
        phone_required=phone_summary is not None,
    )


def reduce_campaign(
        runs: Any,
        *,
        campaign_sha256: str,
        t2_repetitions: int,
) -> dict[str, Any]:
    validate_digest(campaign_sha256, "campaign SHA-256")
    require(isinstance(runs, list), "campaign reduction: runs must be array")
    expected = expected_primary(t2_repetitions)
    require(len(runs) == len(expected),
            "campaign reduction: run count mismatch")
    run_ids = set()
    manifest_digests = set()
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    normalized = []
    for index, (row, expected_row) in enumerate(zip(runs, expected)):
        field = f"campaign run[{index}]"
        require(isinstance(row, dict) and set(row) == RUN_KEYS,
                f"{field}: key set mismatch")
        for key, value in expected_row.items():
            require(row[key] == value,
                    f"{field}: prospective order mismatch")
        run_id = require_string(row["run_id"], f"{field}.run_id")
        require(run_id not in run_ids, f"{field}: duplicate run ID")
        run_ids.add(run_id)
        manifest = validate_digest(
            row["manifest_sha256"], f"{field}.manifest_sha256")
        require(manifest not in manifest_digests,
                f"{field}: duplicate manifest")
        manifest_digests.add(manifest)
        require(
            row["performance_claim_authorized"] is True,
            f"{field}: performance claim is not authorized",
        )
        metrics = _validate_metrics(
            row["metrics"],
            f"{field}.metrics",
            phone_required=row["mode"] in {
                "T1_PHONE_WARM_TIER",
                "T2_PHONE_NO_PROMOTION",
            },
        )
        cells[(row["mode"], row["cache_regime"])].append(metrics)
        normalized.append(row)

    for cell in PRIMARY_CELLS:
        require(len(cells[cell]) == 3,
                f"campaign reduction: primary cell {cell} is incomplete")
    require(
        len(cells[(
            "T2_PHONE_NO_PROMOTION",
            "WARM_HOST_CACHE",
        )]) == t2_repetitions,
        "campaign reduction: T2 cell is incomplete",
    )

    summaries = []
    by_cell = {}
    for (mode, cache_regime), rows in sorted(cells.items()):
        medians = {
            name: _conservative_median([row[name] for row in rows])
            for name in sorted(INTEGER_METRICS)
        }
        zero_completion_count = sum(
            row["completed_request_count"] == 0 for row in rows)
        stranded_repetition_count = sum(
            row["stranded_request_count"] > 0 for row in rows)
        summary = {
            "cache_regime": cache_regime,
            "gpu_energy_scope": "SELECTED_GPU_BOARD_DEVELOPMENT_ONLY",
            "medians": medians,
            "mode": mode,
            "repetitions": len(rows),
            "service_status": (
                "INCLUDES_ZERO_COMPLETION_RUN"
                if zero_completion_count > 0
                else "INCLUDES_STRANDED_REQUESTS"
                if stranded_repetition_count > 0
                else "ALL_REQUESTS_COMPLETED"
            ),
            "stranded_repetition_count": stranded_repetition_count,
            "zero_completion_repetition_count": zero_completion_count,
        }
        summaries.append(summary)
        by_cell[(mode, cache_regime)] = summary

    control = by_cell[(
        "C1_GPU_ONLY_OPTIMIZED",
        "WARM_HOST_CACHE",
    )]["medians"]
    comparisons = []
    for mode in (
            "C2_GPU_PLUS_CPU_WARM_EXECUTOR",
            "T1_PHONE_WARM_TIER"):
        control_cell = by_cell[(
            "C1_GPU_ONLY_OPTIMIZED",
            "WARM_HOST_CACHE",
        )]
        candidate_cell = by_cell[(mode, "WARM_HOST_CACHE")]
        candidate = candidate_cell["medians"]
        service_comparable = (
            control_cell["service_status"] == "ALL_REQUESTS_COMPLETED"
            and candidate_cell["service_status"] == "ALL_REQUESTS_COMPLETED"
        )
        comparisons.append({
            "energy_comparison_status": (
                "DEVELOPMENT_ONLY_EQUAL_WORK"
                if service_comparable else "BLOCKED_UNEQUAL_COMPLETIONS"
            ),
            "mode": mode,
            "selected_gpu_board_energy_ratio_milli":
                _ratio_milli(
                    candidate["selected_gpu_board_energy_nj"],
                    control["selected_gpu_board_energy_nj"],
                ) if service_comparable else None,
            "service_comparable": service_comparable,
            "slo_goodput_ratio_milli":
                _ratio_milli(
                    candidate["slo_goodput_milli_rps"],
                    control["slo_goodput_milli_rps"],
                ),
            "tokens_per_second_ratio_milli":
                _ratio_milli(
                    candidate["tokens_per_second_milli"],
                    control["tokens_per_second_milli"],
                ),
        })
    service_failures = any(
        row["metrics"]["stranded_request_count"] > 0
        for row in normalized
    )
    return {
        "campaign_sha256": campaign_sha256,
        "cells": summaries,
        "comparisons_against_c1_warm": comparisons,
        "energy_claim_authorized": False,
        "energy_scope": "SELECTED_GPU_BOARD_DEVELOPMENT_ONLY",
        "evidence_scope": "MECHANICS_ONLY_CALLER_SUPPLIED",
        "primary_run_count": len(runs),
        "run_bindings": [{
            "manifest_sha256": row["manifest_sha256"],
            "order": row["order"],
            "run_id": row["run_id"],
        } for row in normalized],
        "schema": "s40-primary-campaign-mechanics-v1",
        "status": (
            "MECHANICS_ONLY_COMPLETE_WITH_SERVICE_FAILURES"
            if service_failures else "MECHANICS_ONLY_COMPLETE"
        ),
    }


def reduce_physical_campaign(
        campaign_path: Path,
        manifest_paths: list[Path],
        contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    campaign = read_campaign(campaign_path)
    campaign_sha256 = digest_file(campaign_path)
    require(
        campaign["experiment_contract_sha256"] == digest_file(contract_path),
        "physical campaign: experiment contract mismatch",
    )
    expected = campaign["primary"]
    require(
        isinstance(manifest_paths, list)
        and len(manifest_paths) == len(expected),
        "physical campaign: manifest count mismatch",
    )
    contract = read_json(contract_path, "physical campaign contract")
    requests_path = (
        contract_path.parent / contract["workload"]["requests_path"]
    ).resolve()
    requests_raw = requests_path.read_bytes()
    require(
        digest_bytes(requests_raw) == contract["workload"]["requests_sha256"],
        "physical campaign: request bytes mismatch",
    )

    rows = []
    device_identities: dict[str, tuple[str, str]] = {}
    previous_launch_stop_ns = None
    for index, (manifest_path, campaign_row) in enumerate(
            zip(manifest_paths, expected)):
        field = f"physical campaign run[{index}]"
        require(
            isinstance(manifest_path, Path) and manifest_path.is_file(),
            f"{field}: manifest path is missing",
        )
        before = manifest_path.read_bytes()
        manifest = parse_json(before, f"{field}.manifest")
        require(
            isinstance(manifest, dict)
            and canonical_bytes(manifest) == before,
            f"{field}: manifest is not canonical",
        )
        validation = validate_run_manifest(manifest_path, contract_path)
        require(
            manifest_path.read_bytes() == before,
            f"{field}: manifest changed during validation",
        )
        require(
            validation["status"] == "S40_RUN_MANIFEST_V5_VALID"
            and validation["performance_claim_authorized"] is True,
            f"{field}: physical performance is not authorized",
        )
        binding = manifest.get("campaign_binding")
        require(
            manifest.get("development") is False
            and isinstance(binding, dict)
            and binding == {
                "campaign_id": campaign["campaign_id"],
                "campaign_sha256": campaign_sha256,
                "order": campaign_row["order"],
                "phase": campaign_row["phase"],
            }
            and manifest.get("run_id") == campaign_row["run_id"]
            and manifest.get("mode") == campaign_row["mode"]
            and manifest.get("cache_regime") == campaign_row["cache_regime"]
            and manifest.get("repeat_index")
            == campaign_row["repeat_index"],
            f"{field}: prospective campaign row mismatch",
        )
        snapshots = _snapshot_artifacts(manifest_path, manifest)
        reduced = _reduce_snapshots(snapshots, requests_raw)
        metrics = metrics_from_reduced(
            reduced,
            validation.get("phone_observer_summary"),
        )
        launch = parse_json(
            snapshots["controller_launch"], f"{field}.controller_launch")
        require(
            isinstance(launch, dict)
            and launch.get("run_id") == campaign_row["run_id"],
            f"{field}: launch identity mismatch",
        )
        launch_start_ns = require_int(
            launch.get("started_ns"), f"{field}.launch.started_ns", 1)
        launch_stop_ns = require_int(
            launch.get("stopped_ns"), f"{field}.launch.stopped_ns",
            launch_start_ns,
        )
        if previous_launch_stop_ns is not None:
            require(
                previous_launch_stop_ns <= launch_start_ns,
                f"{field}: physical runs overlap or are reordered",
            )
        previous_launch_stop_ns = launch_stop_ns
        devices = manifest.get("devices")
        require(
            isinstance(devices, list) and devices,
            f"{field}: missing devices",
        )
        current_roles = set()
        for device_index, device in enumerate(devices):
            device_field = f"{field}.devices[{device_index}]"
            require(
                isinstance(device, dict)
                and set(device) == {
                    "boot_id", "device_role", "stable_id"},
                f"{device_field}: key set mismatch",
            )
            role = require_string(
                device["device_role"], f"{device_field}.device_role")
            require(
                role not in current_roles,
                f"{device_field}: duplicate role",
            )
            current_roles.add(role)
            identity = (
                require_string(
                    device["stable_id"], f"{device_field}.stable_id"),
                require_string(
                    device["boot_id"], f"{device_field}.boot_id"),
            )
            if role in device_identities:
                require(
                    identity == device_identities[role],
                    f"{field}: device identity or boot changed for {role}",
                )
            else:
                device_identities[role] = identity
        require("GPU" in current_roles, f"{field}: selected GPU is missing")
        rows.append({
            **campaign_row,
            "manifest_sha256": digest_bytes(before),
            "metrics": metrics,
            "performance_claim_authorized": True,
        })

    mechanics = reduce_campaign(
        rows,
        campaign_sha256=campaign_sha256,
        t2_repetitions=campaign["t2_repetitions"],
    )
    service_failed = (
        mechanics["status"]
        == "MECHANICS_ONLY_COMPLETE_WITH_SERVICE_FAILURES"
    )
    mechanics["evidence_scope"] = "REVALIDATED_PHYSICAL_MANIFESTS"
    mechanics["schema"] = "s40-primary-campaign-summary-v2"
    mechanics["status"] = (
        "MEASURED_COMPLETE_WITH_SERVICE_FAILURES"
        if service_failed else "MEASURED_COMPLETE"
    )
    mechanics["device_identities"] = [
        {
            "boot_id": identity[1],
            "device_role": role,
            "stable_id": identity[0],
        }
        for role, identity in sorted(device_identities.items())
    ]
    mechanics["host_boot_id"] = device_identities["GPU"][1]
    return mechanics
