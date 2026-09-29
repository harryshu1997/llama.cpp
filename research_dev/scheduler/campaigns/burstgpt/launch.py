#!/usr/bin/env python3
"""Resolve, validate, preflight, or execute one BurstGPT campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    RuntimeCapabilityCatalog,
    canonical_json,
    canonical_sha256,
)
from research_dev.scheduler.adapters import (  # noqa: E402
    verify_android_usb_restored,
)
from research_dev.scheduler.config import (  # noqa: E402
    CampaignModelConfiguration,
    ResolvedSchedulerConfiguration,
    SchedulerConfigurationError,
    load_scheduler_configuration,
)
from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import load_helper_evidence  # noqa: E402
from research_dev.scheduler.configuration.campaign import speculative_rows_json  # noqa: E402


CONFIRMATION = "RUN_UNIFIED_FP16_LLAMA_OVERLAY"
COMMAND_MANIFEST_SCHEMA = "research-scheduler-physical-command-manifest-v1"
SOURCE_MANIFEST_SCHEMA = "research-scheduler-source-manifest-v2"


class CampaignLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignLaunchError(message)


def canonical_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("ascii")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def write_new(path: Path, value: object) -> None:
    require(path.is_absolute() and not path.exists(), "output must be new")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _model_for_role(
    configuration: ResolvedSchedulerConfiguration, role: str
) -> CampaignModelConfiguration:
    try:
        return configuration.models.by_trace_role[role]
    except KeyError as error:
        raise CampaignLaunchError(
            "campaign model role is absent: " + role
        ) from error


def _binary(
    configuration: ResolvedSchedulerConfiguration, name: str
) -> Path:
    try:
        return configuration.rig.binaries[name]
    except KeyError as error:
        raise CampaignLaunchError("rig binary is absent: " + name) from error


def _library(
    configuration: ResolvedSchedulerConfiguration, name: str
) -> Path:
    try:
        return configuration.rig.library_directories[name]
    except KeyError as error:
        raise CampaignLaunchError(
            "rig library directory is absent: " + name
        ) from error


def _elastic_phones_json(configuration: ResolvedSchedulerConfiguration) -> str:
    return json.dumps(dict(configuration.campaign.elastic_phones), sort_keys=True, separators=(",", ":"))


def runner_command(
    configuration: ResolvedSchedulerConfiguration,
    *,
    catalog_path: Path,
    source_manifest_path: Path,
    output_path: Path,
    execute: bool,
) -> tuple[str, ...]:
    hot = _model_for_role(configuration, "hot")
    cold = _model_for_role(configuration, "cold")
    overlay = configuration.models.overlay_model
    rig = configuration.rig
    phone = rig.phone
    trace = configuration.campaign.trace
    evidence = configuration.evidence
    command = [
        "python3",
        str(HERE / "runner.py"),
        "--large-requests",
        str(trace.large_requests_path),
        "--overlay-requests",
        str(trace.overlay_requests_path),
        "--trace-manifest",
        str(trace.trace_manifest_path),
        "--source-manifest",
        str(source_manifest_path),
        "--capability-catalog",
        str(catalog_path),
        "--qwen-manifest",
        str(hot.checked_manifest_path),
        "--gemma-manifest",
        str(cold.checked_manifest_path),
        "--qwen-model",
        str(hot.host_artifact_path),
        "--gemma-model",
        str(cold.host_artifact_path),
        "--llama-model",
        str(overlay.host_artifact_path),
        "--gguf-manifest-cache",
        str(configuration.models.manifest_cache_path),
        "--server",
        str(_binary(configuration, "server")),
        "--resident-server",
        str(_binary(configuration, "resident_server")),
        "--cuda-lib-dir",
        str(_library(configuration, "cuda")),
        "--resident-lib-dir",
        str(_library(configuration, "resident")),
        "--bridge",
        str(_binary(configuration, "bridge")),
        "--close-helper",
        str(_binary(configuration, "close_helper")),
        "--adb",
        str(_binary(configuration, "adb")),
        "--phone-usb-close",
        str(_binary(configuration, "phone_usb_close")),
        "--phone-session",
        phone.session_script,
        "--phone-restore",
        phone.restore_script,
        "--phone-worker",
        phone.worker_path,
        "--phone-busybox",
        phone.busybox_path,
        "--qwen-phone-model",
        hot.phone_artifact_path,
        "--gemma-phone-model",
        cold.phone_artifact_path,
        "--llama-phone-model",
        overlay.phone_artifact_path,
        "--phone-session-root",
        phone.session_root,
        "--phone-whole-server",
        phone.whole_server_path,
        "--phone-whole-library-directory",
        phone.whole_library_directory,
        "--phone-whole-model",
        overlay.phone_artifact_path,
        "--phone-whole-state-directory",
        phone.whole_state_directory,
        "--phone-whole-executable-device",
        phone.whole_executable_device,
        *(["--phone-whole-control-transport", phone.whole_control_transport,
           "--phone-whole-ncm-adb-endpoint", phone.whole_ncm_adb_endpoint]
          if phone.whole_control_transport == "adb-ncm" else []),
        "--phone-remote-hash-cache",
        str(phone.remote_hash_cache_path),
        "--phone-diagnostic-endpoint",
        phone.diagnostic_endpoint,
        "--phone-battery-ppm",
        str(phone.battery_ppm),
        "--phone-usb-serial",
        phone.serial,
        "--phone-android-gadget",
        phone.android_gadget_path,
        "--phone-functionfs-gadget",
        phone.functionfs_gadget_path,
        "--phone-functionfs-root",
        phone.functionfs_root_path,
        "--phone-usb-controller",
        phone.usb_controller,
        "--adb-port",
        str(phone.adb_port),
        "--minimum-usb-speed-mbps",
        str(phone.minimum_usb_speed_mbps),
        "--phone-kernel-release",
        phone.kernel_release,
        "--phone-boot-image-sha256",
        phone.boot_image_sha256,
        "--usb-qualification-identity",
        str(evidence.transport_qualification_identity_path),
        "--nmcli",
        str(_binary(configuration, "nmcli")),
        "--selection-mode",
        configuration.campaign.selection_mode,
        "--energy-attribution-kind",
        configuration.campaign.energy_attribution_kind,
        "--output",
        str(output_path),
    ]
    if configuration.campaign.max_workers != 32:
        command.extend((
            "--max-workers", str(configuration.campaign.max_workers)
        ))
    if phone.resident_workers_path is not None:
        assert phone.resident_router_path is not None
        assert phone.multi_session_port_base is not None
        command.extend((
            "--phone-resident-workers", phone.resident_workers_path,
            "--phone-resident-router", phone.resident_router_path,
            "--phone-multi-session-port-base",
            str(phone.multi_session_port_base),
        ))
    for option, model in (
        ("--qwen-ffn-shards", hot),
        ("--gemma-ffn-shards", cold),
        ("--llama-ffn-shards", overlay),
    ):
        if model.phone_ffn_shard_index_path is not None:
            assert model.phone_ffn_shard_directory is not None
            command.extend((
                option,
                str(model.phone_ffn_shard_index_path)
                + "=" + model.phone_ffn_shard_directory,
            ))
    for name, path in rig.transport_host_dependencies.items():
        command.extend((
            "--transport-host-dependency", name + "=" + str(path)
        ))
    if trace.request_indices:
        command.extend((
            "--request-indices",
            ",".join(str(value) for value in trace.request_indices),
        ))
    if trace.arrival_scale is not None:
        command.extend(("--arrival-scale", str(trace.arrival_scale)))
    if trace.replay_schedule_path is not None:
        command.extend((
            "--replay-schedule", str(trace.replay_schedule_path)
        ))
    command.extend((
        "--observation-store-input", str(evidence.observation_store_path)
    ))
    if evidence.observation_source_catalog_path is not None:
        command.extend((
            "--observation-source-catalog",
            str(evidence.observation_source_catalog_path),
        ))
    command.extend((
        "--adaptive-observation-store-input",
        str(evidence.adaptive_observation_store_path),
    ))
    if evidence.adaptive_observation_source_catalog_path is not None:
        command.extend((
            "--adaptive-observation-source-catalog",
            str(evidence.adaptive_observation_source_catalog_path),
        ))
    if configuration.campaign.adaptive_minimum_remaining_tokens is not None:
        command.extend((
            "--adaptive-minimum-remaining-tokens",
            str(configuration.campaign.adaptive_minimum_remaining_tokens),
        ))
    if configuration.campaign.adaptive_maximum_probe_attempts_per_context is not None:
        command.extend((
            "--adaptive-maximum-probe-attempts-per-context",
            str(configuration.campaign.adaptive_maximum_probe_attempts_per_context),
        ))
    if configuration.campaign.adaptive_decode_overrides is not None:
        command.extend((
            "--adaptive-decode-overrides-json",
            json.dumps(dict(configuration.campaign.adaptive_decode_overrides), sort_keys=True),
        ))
    if configuration.campaign.dispatch_policy is not None:
        command.extend((
            "--dispatch-policy-json",
            json.dumps(dict(configuration.campaign.dispatch_policy), sort_keys=True),
        ))
    if configuration.campaign.elastic_phones is not None:
        command.extend(("--elastic-phones-json", _elastic_phones_json(configuration)))
    if configuration.campaign.device_power is not None:
        command.extend(("--device-power-json", canonical_json(configuration.campaign.device_power.to_json())))
    if configuration.campaign.speculative_rows is not None:
        command.extend(("--speculative-rows-json", canonical_json(
            speculative_rows_json(configuration.campaign.speculative_rows))))
    if configuration.campaign.maximum_phone_sessions is not None:
        command.extend((
            "--maximum-phone-sessions",
            str(configuration.campaign.maximum_phone_sessions),
        ))
    if configuration.campaign.host_memory_budget_bytes is not None:
        command.extend((
            "--host-memory-budget-bytes",
            str(configuration.campaign.host_memory_budget_bytes),
        ))
    if configuration.campaign.protected_work_policy != "strict":
        command.extend(("--protected-work-policy", configuration.campaign.protected_work_policy))
    if configuration.campaign.fixed_phone_residency is not None:
        command.extend((
            "--fixed-phone-residency-json",
            canonical_json(configuration.campaign.fixed_phone_residency.to_json()),
        ))
    if configuration.campaign.include_startup_preparation:
        command.append("--include-startup-preparation")
    if configuration.campaign.startup_desktop_parents:
        command.extend(("--startup-desktop-parents-json", json.dumps([
            row.to_json() for row in configuration.campaign.startup_desktop_parents
        ], sort_keys=True, separators=(",", ":"))))
    if configuration.campaign.phone_htp_memory_caps:
        command.extend(("--phone-htp-memory-caps-json", canonical_json([
            cap.to_json() for cap in configuration.campaign.phone_htp_memory_caps
        ])))
    if configuration.campaign.phone_thermal_status_limits:
        command.extend(("--phone-thermal-status-limits-json", canonical_json([
            row.to_json() for row in configuration.campaign.phone_thermal_status_limits
        ])))
    if configuration.campaign.phone_resident_model_reprovisioning is not None:
        command.extend(("--phone-resident-model-reprovisioning-json", canonical_json(
            configuration.campaign.phone_resident_model_reprovisioning.to_json()
        )))
    if configuration.campaign.helper_preparation_fault_injection is not None:
        command.extend((
            "--helper-preparation-fault-injection",
            configuration.campaign.helper_preparation_fault_injection,
        ))
    for path in configuration.evidence.helper_phone_evidence_paths.values():
        command.extend(("--helper-phone-evidence", str(path)))
    if execute:
        command.extend(("--execute", "--confirm", CONFIRMATION))
    return tuple(command)


def offline_residency_command(
    configuration: ResolvedSchedulerConfiguration,
    *,
    catalog_path: Path,
    source_manifest_path: Path,
    output_path: Path,
    execute: bool,
) -> tuple[str, ...]:
    command = list(runner_command(
        configuration,
        catalog_path=catalog_path,
        source_manifest_path=source_manifest_path,
        output_path=output_path,
        execute=execute,
    ))
    command[1] = str(HERE / "offline_residency_gate.py")
    return tuple(command)


def desktop_parent_calibration_command(
    configuration: ResolvedSchedulerConfiguration,
    *,
    catalog_path: Path,
    source_manifest_path: Path,
    output_path: Path,
    execute: bool,
    desktop_parent_role: str = "cold",
) -> tuple[str, ...]:
    command = list(runner_command(
        configuration,
        catalog_path=catalog_path,
        source_manifest_path=source_manifest_path,
        output_path=output_path,
        execute=execute,
    ))
    command[1] = str(HERE / "desktop_parent_calibration.py")
    command.extend((
        "--desktop-baseline-plans",
        str(configuration.models.desktop_baseline_plans_path),
        "--desktop-parent-role",
        desktop_parent_role,
    ))
    return tuple(command)


def preflight_command(
    configuration: ResolvedSchedulerConfiguration,
    *,
    catalog_path: Path,
    normal_usb_receipt_path: Path,
    output_path: Path,
) -> tuple[str, ...]:
    hot = _model_for_role(configuration, "hot")
    cold = _model_for_role(configuration, "cold")
    overlay = configuration.models.overlay_model
    trace = configuration.campaign.trace
    evidence = configuration.evidence
    phone = configuration.rig.phone
    command = [
        "python3",
        str(HERE / "preflight.py"),
        "--large-requests", str(trace.large_requests_path),
        "--overlay-requests", str(trace.overlay_requests_path),
        "--trace-manifest", str(trace.trace_manifest_path),
        "--capability-catalog", str(catalog_path),
        "--observation-store-input", str(evidence.observation_store_path),
        "--adaptive-observation-store-input",
        str(evidence.adaptive_observation_store_path),
        "--qwen-manifest", str(hot.checked_manifest_path),
        "--gemma-manifest", str(cold.checked_manifest_path),
        "--qwen-model", str(hot.host_artifact_path),
        "--gemma-model", str(cold.host_artifact_path),
        "--llama-model", str(overlay.host_artifact_path),
        "--server", str(_binary(configuration, "server")),
        "--resident-server",
        str(_binary(configuration, "resident_server")),
        "--cuda-lib-dir", str(_library(configuration, "cuda")),
        "--resident-lib-dir", str(_library(configuration, "resident")),
        "--bridge", str(_binary(configuration, "bridge")),
        "--adb", str(_binary(configuration, "adb")),
        "--phone-session-discovery",
        str(evidence.phone_session_discovery_path),
        "--close-helper", str(_binary(configuration, "close_helper")),
        "--phone-diagnostic-endpoint", phone.diagnostic_endpoint,
        "--phone-battery-ppm", str(phone.battery_ppm),
        "--phone-usb-serial", phone.serial,
        "--phone-normal-usb-receipt", str(normal_usb_receipt_path),
        "--adb-port", str(phone.adb_port),
        "--minimum-usb-speed-mbps", str(phone.minimum_usb_speed_mbps),
        "--output", str(output_path),
    ]
    if trace.replay_schedule_path is not None:
        command.extend(("--replay-schedule", str(trace.replay_schedule_path)))
    for option, model in (
        ("--qwen-ffn-shards", hot),
        ("--gemma-ffn-shards", cold),
        ("--llama-ffn-shards", configuration.models.overlay_model),
    ):
        if model.phone_ffn_shard_index_path is not None:
            assert model.phone_ffn_shard_directory is not None
            command.extend((
                option,
                str(model.phone_ffn_shard_index_path)
                + "=" + model.phone_ffn_shard_directory,
            ))
    for helper in configuration.rig.helper_phones:
        command.extend(("--helper-phone", json.dumps(helper.to_json(), sort_keys=True)))
    for path in configuration.evidence.helper_phone_evidence_paths.values():
        command.extend(("--helper-phone-evidence", str(path)))
    if configuration.campaign.elastic_phones is not None:
        command.extend(("--elastic-phones-json", _elastic_phones_json(configuration)))
    if configuration.campaign.device_power is not None:
        command.extend(("--device-power-json", canonical_json(configuration.campaign.device_power.to_json())))
    if configuration.campaign.speculative_rows is not None:
        command.extend(("--speculative-rows-json", canonical_json(
            speculative_rows_json(configuration.campaign.speculative_rows))))
    return tuple(command)


def _source_manifest(
    configuration: ResolvedSchedulerConfiguration,
) -> dict[str, object]:
    root = configuration.rig.repo_root
    scheduler = root / "research_dev" / "scheduler"
    rows = [
        {
            "path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
        }
        for path in sorted(scheduler.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".sh", ".zsh"}
    ]

    def git(*arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            encoding="ascii",
            text=True,
        ).stdout.strip()

    body: dict[str, object] = {
        "branch": git("branch", "--show-current"),
        "files": rows,
        "head": git("rev-parse", "HEAD"),
        "resolved_configuration": configuration.to_json(),
        "schema": SOURCE_MANIFEST_SCHEMA,
    }
    source = configuration.evidence.prematerialized_source_manifest_path
    if source is not None:
        body["prematerialized_source_manifest_sha256"] = file_sha256(source)
    body["manifest_sha256"] = canonical_sha256(body)
    return body


def _validate_required_paths(
    configuration: ResolvedSchedulerConfiguration,
) -> None:
    paths: list[Path] = [
        configuration.rig.repo_root,
        configuration.models.desktop_baseline_plans_path,
        configuration.evidence.kernel_profile_path,
        configuration.evidence.calibration_directory,
        configuration.evidence.phone_session_discovery_path,
        configuration.evidence.transport_qualification_identity_path,
        configuration.evidence.observation_store_path,
        configuration.evidence.adaptive_observation_store_path,
        configuration.campaign.trace.large_requests_path,
        configuration.campaign.trace.overlay_requests_path,
        configuration.campaign.trace.trace_manifest_path,
        *configuration.rig.binaries.values(),
        *configuration.rig.library_directories.values(),
        *configuration.rig.transport_host_dependencies.values(),
        *configuration.evidence.transport_qualification_directories,
        *configuration.evidence.helper_phone_evidence_paths.values(),
        *(row.host_artifact_path for row in configuration.models.models),
        *(
            row.phone_ffn_shard_index_path
            for row in configuration.models.models
            if row.phone_ffn_shard_index_path is not None
        ),
        *(
            row.checked_manifest_path
            for row in configuration.models.models
            if row.checked_manifest_path is not None
        ),
    ]
    optional = (
        configuration.evidence.prematerialized_catalog_path,
        configuration.evidence.prematerialized_source_manifest_path,
        configuration.evidence.overlay_catalog_path,
        configuration.evidence.observation_source_catalog_path,
        configuration.evidence.adaptive_observation_source_catalog_path,
        configuration.campaign.trace.replay_schedule_path,
    )
    paths.extend(path for path in optional if path is not None)
    missing = tuple(str(path) for path in paths if not path.exists())
    require(not missing, "configuration paths are absent: " + ",".join(missing))


def _materialize_catalog(
    configuration: ResolvedSchedulerConfiguration,
    campaign_path: Path,
    output_path: Path,
) -> None:
    source = configuration.evidence.prematerialized_catalog_path
    if source is not None:
        shutil.copyfile(source, output_path)
    else:
        subprocess.run(
            (
                "python3",
                str(HERE / "catalog.py"),
                "--campaign",
                str(campaign_path),
                "--output",
                str(output_path),
            ),
            cwd=configuration.rig.repo_root,
            check=True,
        )
    catalog = RuntimeCapabilityCatalog.from_json(json.loads(
        output_path.read_text(encoding="ascii")
    ))
    require(
        catalog.maximum_latency_ppm
            == configuration.campaign.maximum_latency_ppm,
        "catalog latency policy differs from campaign",
    )
    power = configuration.evidence.phone_power
    helper_power = {device: load_helper_evidence(path).power
                    for device, path in configuration.evidence.helper_phone_evidence_paths.items()}
    profiles = tuple(row for row in catalog.phone_power_profiles if row.device_id not in helper_power)
    require({row.device_id: row for row in catalog.phone_power_profiles if row.device_id in helper_power}
            == helper_power, "catalog helper phone power differs from qualified evidence")
    if power.allow_assumed_for_scheduling:
        require(
            len(profiles) == 1
            and profiles[0].evidence_kind == power.evidence_kind
            and profiles[0].active_power_mw == power.active_power_mw
            and profiles[0].idle_power_mw == power.idle_power_mw
            and profiles[0].allow_assumed_for_scheduling,
            "catalog phone power differs from evidence manifest",
        )


def normalized_runner_contract(command: Iterable[str]) -> dict[str, object]:
    values = list(command)
    require(len(values) >= 2, "runner command is empty")
    result: dict[str, object] = {
        "interpreter": values[0],
        "runner": Path(values[1]).name,
    }
    repeated = {"--transport-host-dependency", "--helper-phone-evidence"}
    index = 2
    while index < len(values):
        option = values[index]
        require(option.startswith("--"), "runner command option is invalid")
        if option in {"--execute", "--include-startup-preparation"}:
            result[option] = True
            index += 1
            continue
        require(index + 1 < len(values), "runner command option lacks a value")
        value = values[index + 1]
        if option in repeated:
            result.setdefault(option, [])
            result[option].append(value)
        else:
            require(option not in result, "runner command option is duplicated")
            result[option] = value
        index += 2
    for option in ("--output", "--source-manifest", "--capability-catalog"):
        if option in result:
            result[option] = "<campaign-output>"
    return result


def command_manifest(
    configuration: ResolvedSchedulerConfiguration,
    command: tuple[str, ...],
    catalog_path: Path,
) -> dict[str, object]:
    contract = normalized_runner_contract(command)
    unsigned = {
        "catalog_sha256": file_sha256(catalog_path),
        "configuration_sha256": configuration.configuration_sha256,
        "normalized_runner_contract": contract,
        "runner_contract_sha256": canonical_sha256(contract),
        "schema": COMMAND_MANIFEST_SCHEMA,
    }
    return {**unsigned, "manifest_sha256": canonical_sha256(unsigned)}


def _run_streamed(
    command: tuple[str, ...], *, cwd: Path, log_path: Path
) -> None:
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for block in iter(lambda: process.stdout.read(65536), b""):
            log.write(block)
            sys.stdout.buffer.write(block)
            sys.stdout.buffer.flush()
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def launch(
    campaign_path: Path,
    output: Path,
    *,
    preflight_only: bool,
    resolve_only: bool,
    offline_residency_only: bool = False,
    desktop_parent_calibration_only: bool = False,
    desktop_parent_role: str = "cold",
) -> int:
    require(
        campaign_path.is_file(), "campaign manifest is absent"
    )
    require(
        output.is_absolute() and not output.exists(),
        "output must be a new absolute path",
    )
    configuration = load_scheduler_configuration(campaign_path)
    require(
        not configuration.rig.helper_phones or preflight_only or resolve_only
        or {row.device_id for row in configuration.rig.helper_phones}
        == set(configuration.evidence.helper_phone_evidence_paths),
        "rig helper phones need qualified evidence and a campaign lifecycle",
    )
    _validate_required_paths(configuration)
    active = subprocess.run(
        (
            "pgrep", "-f",
            "[r]esearch_dev/scheduler/campaigns/burstgpt/"
            "(runner|offline_residency_gate|desktop_parent_calibration).py",
        ),
        check=False,
        capture_output=True,
    )
    require(active.returncode == 1, "another physical campaign is active")
    output.mkdir(parents=True)
    (output / "RESOLVED_CONFIGURATION.json").write_bytes(
        configuration.canonical_bytes()
    )
    source_manifest_path = output / "SOURCE_MANIFEST.json"
    source_manifest_path.write_bytes(canonical_bytes(
        _source_manifest(configuration)
    ))
    catalog_path = output / "UNIFIED_RUNTIME_CATALOG.json"
    _materialize_catalog(
        configuration, campaign_path.resolve(), catalog_path
    )
    run_output = output / "run"
    command_builder = (
        desktop_parent_calibration_command
        if desktop_parent_calibration_only else
        offline_residency_command
        if offline_residency_only else
        runner_command
    )
    command = command_builder(
        configuration,
        catalog_path=catalog_path,
        source_manifest_path=source_manifest_path,
        output_path=run_output,
        execute=True,
        **(
            {"desktop_parent_role": desktop_parent_role}
            if desktop_parent_calibration_only else {}
        ),
    )
    (output / "COMMAND_MANIFEST.json").write_bytes(canonical_bytes(
        command_manifest(configuration, command, catalog_path)
    ))
    (output / "RUN_COMMAND.txt").write_text(
        shlex.join(command) + "\n", encoding="ascii"
    )
    if resolve_only:
        return 0
    if preflight_only:
        normal_usb_path = output / "PHONE_USB_BEFORE.json"
        normal_usb = verify_android_usb_restored(
            serial=configuration.rig.phone.serial,
            adb_port=configuration.rig.phone.adb_port,
            minimum_speed_mbps=(
                configuration.rig.phone.minimum_usb_speed_mbps
            ),
            timeout_s=60,
        )
        normal_usb_path.write_bytes(canonical_bytes(normal_usb.to_json()))
        preflight_output = output / "PHYSICAL_PREFLIGHT.json"
        command = preflight_command(
            configuration,
            catalog_path=catalog_path,
            normal_usb_receipt_path=normal_usb_path,
            output_path=preflight_output,
        )
        _run_streamed(
            command,
            cwd=configuration.rig.repo_root,
            log_path=output / "preflight.log",
        )
        return 0
    _run_streamed(
        command,
        cwd=configuration.rig.repo_root,
        log_path=output / "runner.log",
    )
    if desktop_parent_calibration_only:
        result_path = run_output / "RESULT.json"
        calibration_path = run_output / "DESKTOP_PARENT_CALIBRATION.json"
        plans_path = run_output / "MEASURED_DESKTOP_BASELINE_PLANS_V1.json"
        require(
            result_path.is_file()
            and calibration_path.is_file()
            and plans_path.is_file(),
            "desktop parent calibration result is incomplete",
        )
        sums = {
            str(path.relative_to(output)): file_sha256(path)
            for path in (
                catalog_path,
                source_manifest_path,
                output / "RESOLVED_CONFIGURATION.json",
                output / "COMMAND_MANIFEST.json",
                output / "RUN_COMMAND.txt",
                result_path,
                calibration_path,
                plans_path,
            )
        }
        (output / "SHA256SUMS.json").write_bytes(canonical_bytes(sums))
        return 0
    result_path = run_output / "RESULT.json"
    journal_path = run_output / "SCHEDULER_DECISION_LOG.json"
    require(
        result_path.is_file() and journal_path.is_file(),
        "physical campaign result is incomplete",
    )
    sums = {
        str(path.relative_to(output)): file_sha256(path)
        for path in (
            catalog_path,
            source_manifest_path,
            output / "RESOLVED_CONFIGURATION.json",
            output / "COMMAND_MANIFEST.json",
            output / "RUN_COMMAND.txt",
            result_path,
            journal_path,
        )
    }
    (output / "SHA256SUMS.json").write_bytes(canonical_bytes(sums))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--resolve-only", action="store_true")
    modes.add_argument("--offline-residency-only", action="store_true")
    modes.add_argument(
        "--desktop-parent-calibration-only", action="store_true"
    )
    parser.add_argument(
        "--desktop-parent-role", choices=("cold", "hot"), default="cold"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return launch(
            args.campaign,
            args.output,
            preflight_only=args.preflight_only,
            resolve_only=args.resolve_only,
            offline_residency_only=args.offline_residency_only,
            desktop_parent_calibration_only=(
                args.desktop_parent_calibration_only
            ),
            desktop_parent_role=args.desktop_parent_role,
        )
    except (CampaignLaunchError, SchedulerConfigurationError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
