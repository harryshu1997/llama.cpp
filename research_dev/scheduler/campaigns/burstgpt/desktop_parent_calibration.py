#!/usr/bin/env python3
"""Calibrate one live-VRAM-selected desktop parent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from urllib.parse import urlsplit


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    DeviceMemoryCapacity,
    RuntimePlacementSnapshot,
)
from research_dev.scheduler.adapters import (  # noqa: E402
    default_host_metric_callbacks,
    HostEnergySampler,
    LlamaCppCompletionPayload,
    LlamaCppHttpClient,
    LlamaServerLaunchContract,
    LlamaServerProcessConfiguration,
    LlamaServerProcessLauncher,
    nvidia_gpu_snapshot,
    probe_nvidia_process_memory_bytes,
    server_energy_summary,
)
from research_dev.scheduler.campaigns.burstgpt import runner  # noqa: E402
from research_dev.scheduler.adapters.llama_server import llama_server_runtime_timing  # noqa: E402


SCHEMA = "s42-capacity-aware-desktop-parent-calibration-v1"


class DesktopParentCalibrationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DesktopParentCalibrationError(message)


def _canonical(value: object) -> bytes:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _parse_ldd_libraries(text: str, root: Path) -> dict[str, Path]:
    """Shared libraries that ``ldd`` resolved to real files inside ``root``: name -> real file."""
    found: dict[str, Path] = {}
    for line in text.splitlines():
        name, separator, tail = line.partition("=>")
        if not separator:
            continue
        resolved = tail.strip().split(" (", 1)[0].strip()
        if not resolved or resolved == "not found":
            continue
        try:
            real = Path(resolved).resolve(strict=True)
        except OSError:
            continue
        if root in real.parents:
            found[name.strip()] = real
    return found


def _runtime_libraries_sha256(binary: Path) -> dict[str, object]:
    """Digests of every shared library ``ldd`` resolves inside the binary's own directory tree.

    ``llama-server`` is an 18 KB dynamically linked launcher; the loader, graph and server code
    live in ``libllama.so`` and ``libllama-server-impl.so`` next to it, so the executable digest
    alone does not identify the build that ran. Libraries outside the build tree (CUDA, libc)
    are not recorded. ``ldd_error`` is set when ``ldd`` could not run.
    """
    binary = binary.resolve()
    try:
        completed = subprocess.run(
            ["ldd", str(binary)], capture_output=True, text=True, check=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"libraries": {}, "ldd_error": type(error).__name__ + ": " + str(error)}
    libraries = _parse_ldd_libraries(completed.stdout, binary.parent)
    return {
        "libraries": {real.name: _sha256(real) for _name, real in sorted(libraries.items())},
        "ldd_error": None,
    }


def _write_new(path: Path, value: object) -> None:
    require(not path.exists(), "calibration artifact already exists")
    path.write_bytes(_canonical(value))


def _wait_for_sampler(sampler: HostEnergySampler, count: int) -> None:
    deadline = time.monotonic() + 10
    while len(sampler.rows()) < count:
        if time.monotonic() >= deadline:
            raise DesktopParentCalibrationError(
                "host energy sampler did not provide coverage"
            )
        time.sleep(0.05)


def _endpoint_is_free(endpoint: str) -> bool:
    parsed = urlsplit(endpoint)
    require(
        parsed.hostname in {"127.0.0.1", "localhost"}
        and parsed.port is not None,
        "calibration endpoint is not local",
    )
    probe = socket.socket()
    probe.settimeout(0.2)
    try:
        return probe.connect_ex((parsed.hostname, parsed.port)) != 0
    finally:
        probe.close()


def _calibration_request(
    models: object,
    *,
    role: str,
    context_size: int,
    selected_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    rows = [
        row for row in (models.large_rows if selected_rows is None else selected_rows)
        if runner.trace_role(row) == role
        and int(row["input_tokens"]) + int(row["output_tokens"])
            <= context_size
    ]
    require(bool(rows), "desktop parent calibration request is absent")
    return max(rows, key=lambda row: (
        int(row["output_tokens"]),
        -int(row["input_tokens"]),
        -int(row["request_index"]),
    ))


def _payload(
    row: dict[str, object],
    output: Path,
    *,
    request_id: str,
    model_alias: str,
) -> LlamaCppCompletionPayload:
    return LlamaCppCompletionPayload(
        request_id=request_id,
        expected_model_alias=model_alias,
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        prompt_tokens=tuple(int(value) for value in row["prompt_tokens"]),
        seed=42,
        stream_path=output / (request_id + ".raw"),
        on_first_token=lambda _value: None,
        quality_mode="semantic",
        timeout_s=3_600,
    )


def _measurement(
    sampler: HostEnergySampler,
    start_ns: int,
    end_ns: int,
) -> dict[str, object]:
    rows = sampler.rows_between(start_ns, end_ns)
    return {
        "duration_us": (end_ns - start_ns) // 1000,
        "energy": dict(server_energy_summary(rows, start_ns, end_ns)),
        "sample_count": len(rows),
    }


def _replace_desktop_plan(
    source_path: Path,
    artifact_sha256: str,
    *,
    gpu_first_layer: int,
    placement_sha256: str,
    evidence_id: str,
    source_placement_sha256: str,
    maximum_gpu_layers: int,
    live_free_vram_bytes: int,
    required_with_reserve_bytes: int,
    peak_vram_bytes: int,
    gpu_device_id: str,
    gpu_weight_allocation_ppm: int,
    cuda_graph_mode: str = "default",
    launch_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    value = json.loads(source_path.read_text(encoding="ascii"))
    require(
        type(value) is dict
        and value.get("schema")
            == "s42-measured-desktop-baseline-plans-v1"
        and type(value.get("plans")) is list,
        "source desktop baseline plans are invalid",
    )
    replaced = False
    plans = []
    for raw in value["plans"]:
        row = dict(raw)
        if row.get("artifact_sha256") == artifact_sha256:
            parameters = dict(row.get("adapter_parameters", {}))
            parameters.update({
                "cuda_graph_mode": cuda_graph_mode,
                "capacity_parent_calibration_live_free_bytes": (
                    live_free_vram_bytes
                ),
                "capacity_parent_maximum_gpu_layers": maximum_gpu_layers,
                "capacity_parent_peak_vram_bytes": peak_vram_bytes,
                "capacity_parent_qualification_sha256": evidence_id,
                "capacity_parent_required_with_reserve_bytes": (
                    required_with_reserve_bytes
                ),
                "capacity_parent_source_placement_sha256": (
                    source_placement_sha256
                ),
                "memory_model_weight_allocation_ppm:"
                    + gpu_device_id: gpu_weight_allocation_ppm,
            })
            parameters.update(launch_overrides or {})
            row.update({
                "adapter_parameters": parameters,
                "cuda_graph_mode": cuda_graph_mode,
                "desktop_placement_sha256": placement_sha256,
                "evidence_ids": [evidence_id],
                "gpu_first_layer": gpu_first_layer,
                "identity_evidence_id": evidence_id,
                "maturity": "QUALIFIED",
            })
            replaced = True
        plans.append(row)
    require(replaced, "desktop baseline plan is absent")
    return {**value, "plans": plans}


def run(args: argparse.Namespace) -> dict[str, object]:
    require(
        args.desktop_baseline_plans.is_file(),
        "desktop baseline plan input is absent",
    )
    models = runner._load_trace_models(args)
    scheduler, manifests, _ = runner._build_scheduler(
        args, models, load_adaptive_observations=False
    )
    expected_by_role = {
        runner.GEMMA_ROLE: models.expected_gemma,
        runner.QWEN_ROLE: models.expected_qwen,
    }
    expected = expected_by_role[args.desktop_parent_role]
    manifest = manifests[expected.model_id]
    catalog = models.catalog
    control = catalog.desktop_control_by_artifact[manifest.artifact_sha256]
    source = catalog.composite_executor_by_id[control.executor_id]
    gpu_device_id = str(source.adapter_parameters["gpu_device_id"])
    gpu_resource_id = catalog.placement_profile.devices[
        gpu_device_id
    ].memory_pool_id
    gpu = nvidia_gpu_snapshot()
    reserve_bytes = min(512 * 1024**2, int(gpu["memory_free_bytes"]))
    memory = RuntimePlacementSnapshot(
        snapshot_id="desktop-parent-calibration-live-nvml",
        captured_at_us=0,
        valid_until_us=60_000_000,
        capacities={
            gpu_resource_id: DeviceMemoryCapacity(
                gpu_resource_id,
                int(gpu["memory_total_bytes"]),
                int(gpu["memory_total_bytes"])
                    - int(gpu["memory_free_bytes"]),
                reserve_bytes,
            ),
        },
    )
    launch_overrides = {
        name: getattr(args, name)
        for name in ("context_size", "parallel", "batch_size", "ubatch_size", "desktop_launch_mode")
        if getattr(args, name, None) is not None
    }
    selection = scheduler.select_live_vram_desktop_parent(
        manifest.model_id, memory,
        cuda_graph_mode=args.cuda_graph_mode,
        preserve_placement=args.preserve_desktop_placement,
        maximum_gpu_layers=getattr(args, "maximum_gpu_layers", None),
        launch_overrides=launch_overrides,
    )
    selected = selection.selected
    current_gpu = nvidia_gpu_snapshot()
    require(
        int(current_gpu["memory_free_bytes"])
            >= selected.required_with_reserve_bytes,
        "live VRAM changed after desktop parent selection",
    )
    endpoint = source.endpoint
    require(_endpoint_is_free(endpoint), "calibration endpoint is already active")
    parameters = selected.adapter_parameters
    contract = LlamaServerLaunchContract(
        model_alias=str(parameters["model_alias"]),
        context_size=int(parameters["context_size"]),
        parallel=int(parameters["parallel"]),
        batch_size=int(parameters["batch_size"]),
        ubatch_size=int(parameters["ubatch_size"]),
        gpu_layers=selected.gpu_layers,
        cpu_device_id=str(parameters["cpu_device_id"]),
        gpu_device_id=gpu_device_id,
        phone_device_id=None,
        ffn_environment={},
        cuda_graph_mode=parameters.get("cuda_graph_mode", "default"),
        desktop_launch_mode=parameters.get("desktop_launch_mode", "canonical"),
        threads=int(parameters.get("threads", 0)),
        threads_batch=int(parameters.get("threads_batch", 0)),
        cpu_affinity=(
            None if "cpu_affinity" not in parameters
            else str(parameters["cpu_affinity"])
        ),
    )
    launcher = LlamaServerProcessLauncher(
        LlamaServerProcessConfiguration(
            server_path=args.server,
            model_paths_by_artifact={
                manifest.artifact_sha256: models.model_paths[manifest.model_id]
            },
            library_paths_by_device={
                gpu_device_id: (args.cuda_lib_dir,)
            },
            executable_device_names={gpu_device_id: "CUDA0"},
            output_directory=args.output,
            common_library_paths=(args.server.parent, args.cuda_lib_dir),
        )
    )
    sampler = HostEnergySampler(default_host_metric_callbacks(), interval_s=0.1)
    client = LlamaCppHttpClient()
    _, selected_requests, _, _ = runner._select_replay(args, models, manifests)
    row = _calibration_request(
        models,
        role=args.desktop_parent_role,
        context_size=contract.context_size,
        selected_rows=[item["row"] for item in selected_requests if item["source"] == "large"],
    )
    process = None
    sampler.start()
    try:
        _wait_for_sampler(sampler, 2)
        load_start_ns = time.monotonic_ns()
        process = launcher.launch_contract(
            endpoint,
            contract,
            manifest,
            label="capacity-parent-" + args.desktop_parent_role,
            control_check=lambda: None,
        )
        load_end_ns = time.monotonic_ns()
        cold_payload = _payload(
            row,
            args.output,
            request_id="capacity-parent-cold",
            model_alias=contract.model_alias,
        )
        cold_start_ns = time.monotonic_ns()
        cold_log_index = len(process.stderr_lines)
        cold = client.complete(endpoint, cold_payload, lambda: None)
        cold_end_ns = time.monotonic_ns()
        hot_log_index = len(process.stderr_lines)
        hot_payload = _payload(
            row,
            args.output,
            request_id="capacity-parent-hot",
            model_alias=contract.model_alias,
        )
        hot_start_ns = time.monotonic_ns()
        hot = client.complete(endpoint, hot_payload, lambda: None)
        hot_end_ns = time.monotonic_ns()
        hot_log_end = len(process.stderr_lines)
        process_vram_samples = [
            probe_nvidia_process_memory_bytes(process.pid)
        ]
        time.sleep(0.3)
        _wait_for_sampler(sampler, 3)
        rows = sampler.rows()
        process_vram_samples.append(
            probe_nvidia_process_memory_bytes(process.pid)
        )
        process_vram = max(
            value for value in process_vram_samples if value is not None
        )
        live_gpu = nvidia_gpu_snapshot()
        baseline_vram = int(gpu["memory_used_bytes"])
        peak_total_vram = max(
            int(sample["gpu"]["memory_used_bytes"]) for sample in rows
        )
        peak_total_vram = max(
            peak_total_vram,
            int(live_gpu["memory_used_bytes"]),
            baseline_vram + process_vram,
        )
        result = {
            "cuda_graph_mode": contract.cuda_graph_mode,
            "launch_contract": {
                "cuda_graph_mode": contract.cuda_graph_mode,
                "desktop_launch_mode": contract.desktop_launch_mode,
                "context_size": contract.context_size,
                "parallel": contract.parallel,
                "batch_size": contract.batch_size,
                "ubatch_size": contract.ubatch_size,
                "gpu_layers": contract.gpu_layers,
                "threads": contract.threads,
                "threads_batch": contract.threads_batch,
                "cpu_affinity": contract.cpu_affinity,
            },
            "artifact_bytes": manifest.artifact_bytes,
            "artifact_sha256": manifest.artifact_sha256,
            "baseline_gpu": gpu,
            "cold_request": {
                "runtime_timing": llama_server_runtime_timing(
                    process.stderr_lines[cold_log_index:hot_log_index]
                ),
                **_measurement(sampler, cold_start_ns, cold_end_ns),
                "request_index": row["request_index"],
                "result": cold,
            },
            "hot_request": {
                "runtime_timing": llama_server_runtime_timing(
                    process.stderr_lines[hot_log_index:hot_log_end]
                ),
                **_measurement(sampler, hot_start_ns, hot_end_ns),
                "request_index": row["request_index"],
                "result": hot,
            },
            "model_load": _measurement(sampler, load_start_ns, load_end_ns),
            "peak_process_vram_bytes": process_vram,
            "peak_total_vram_bytes": peak_total_vram,
            "peak_vram_delta_bytes": max(0, peak_total_vram - baseline_vram),
            "placement_sha256": selected.placement_sha256,
            "trace_role": args.desktop_parent_role,
            "runtime_binary_sha256": _sha256(args.server),
            "runtime_libraries_sha256": _runtime_libraries_sha256(args.server),
            "schema": SCHEMA,
            "selection": selection.to_json(),
            "status": "PASS",
        }
    finally:
        if process is not None:
            process.stop()
        sampler.stop()
    result_path = args.output / "DESKTOP_PARENT_CALIBRATION.json"
    _write_new(result_path, result)
    evidence_id = _sha256(result_path)
    plans = _replace_desktop_plan(
        args.desktop_baseline_plans,
        manifest.artifact_sha256,
        gpu_first_layer=selected.gpu_first_layer,
        placement_sha256=selected.placement_sha256,
        evidence_id=evidence_id,
        source_placement_sha256=selection.source_placement_sha256,
        maximum_gpu_layers=selection.maximum_gpu_layers,
        live_free_vram_bytes=selected.live_free_vram_bytes,
        required_with_reserve_bytes=selected.required_with_reserve_bytes,
        peak_vram_bytes=int(result["peak_process_vram_bytes"]),
        gpu_device_id=gpu_device_id,
        gpu_weight_allocation_ppm=(
            selected.gpu_weight_allocation_ppm
        ),
        cuda_graph_mode=contract.cuda_graph_mode,
        launch_overrides=launch_overrides,
    )
    _write_new(
        args.output / "MEASURED_DESKTOP_BASELINE_PLANS_V1.json",
        plans,
    )
    summary = {
        "cuda_graph_mode": contract.cuda_graph_mode,
        "calibration_sha256": evidence_id,
        "gpu_layers": selected.gpu_layers,
        "placement_sha256": selected.placement_sha256,
        "status": "PASS",
        "trace_role": args.desktop_parent_role,
    }
    _write_new(args.output / "RESULT.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = runner._build_parser()
    parser.add_argument(
        "--desktop-baseline-plans", type=Path, required=True
    )
    parser.add_argument(
        "--desktop-parent-role",
        choices=(runner.GEMMA_ROLE, runner.QWEN_ROLE),
        default=runner.GEMMA_ROLE,
    )
    parser.add_argument("--cuda-graph-mode", choices=("default", "disabled"))
    parser.add_argument("--preserve-desktop-placement", action="store_true")
    parser.add_argument("--maximum-gpu-layers", type=int)
    parser.add_argument("--desktop-launch-mode", choices=("canonical", "runtime-defaults"))
    for name in ("context-size", "parallel", "batch-size", "ubatch-size"):
        parser.add_argument("--" + name, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        runner._validate_arguments(args)
        args.output.mkdir(parents=True)
        result = run(args)
    except BaseException as error:
        if args.output.is_dir():
            failure = args.output / "FAILURE.json"
            if not failure.exists():
                failure.write_bytes(_canonical({
                    "error": type(error).__name__ + ": " + str(error),
                    "schema": "s42-desktop-parent-calibration-failure-v1",
                    "status": "FAIL",
                    "trace_executed": False,
                    "traceback": traceback.format_exc(),
                }))
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
