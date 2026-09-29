#!/usr/bin/env python3
"""Build a unified plan for an oversized Gemma CUDA/CPU/OP15 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    ApplicabilityContract,
    BackendArtifact,
    CandidateSet,
    CapacityPlanner,
    CohortPolicy,
    CohortPlanner,
    DeviceMemoryCapacity,
    LayerPlacementCandidate,
    MemoryRequirement,
    MetricEstimate,
    ModelIdentity,
    OperatorOffloadContract,
    OperatorSplitPolicy,
    OverlapEstimate,
    PhoneBackendConfig,
    PhoneTransportContract,
    PlacementGranularity,
    QualityClass,
    QualityContract,
    ResourceRequirement,
    RouteAlternative,
    RouteMaturity,
    SchedulingUnit,
    UnitKind,
    build_execution_plan,
    canonical_sha256,
    make_work_set_hash,
    write_execution_plan,
)
from research_dev.spikes.s42_general_energy_scheduler_v1.full_fp16_burstgpt_v1.shape_balance_v1.policy_adapter import (  # noqa: E402
    DEFAULT_CALIBRATION,
    VARIANTS as SPLIT_POLICY_VARIANTS,
    materialize_gemma_policy,
)


PROFILE_ID = "burstgpt-gemma-f16-overflow-4060ti-op15-v1"
WORKLOAD_ID = "burstgpt-source-cold17-gemma-f16-overflow-v1"
UNIT_ID = "burstgpt-source-cold17-v1"
SOURCE_COLD_MODEL = "qwen3-14b-q4_k_m"
TARGET_MODEL_ID = "gemma-4-12b-f16-storage-proxy"
MODEL_BYTES = 23_832_065_056
MODEL_SHA256 = (
    "ed76f2183d2d1d65091986033023e6c7"
    "8d27f6276c1b0c5826cc92acf73538cf"
)
TRACE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4"
    "c31a4a9e8b4f9c085b29a14f80250a0ff"
)
COLD_WORK_SHA256 = (
    "3e299c2e574a49fad474f81f5cc3bd2e"
    "c34e7a8e2bb50eac1bcda6a5b0876c10"
)
CONTROL_ROUTE = "gemma-f16-cuda25-cpu23-control-v1"
OFFLOAD_ROUTE = "gemma-f16-cuda25-cpu23-op15-ffn-v1"
CONTROL_RESULT_SHA256 = (
    "97f701102d2136316b4c557dbf811fd9"
    "ee59fd0e4501c839b32a8fd507696147"
)
OFFLOAD_RESULT_SHA256 = (
    "a307eb6cb4bc80042e07ab79ca4fc9b5"
    "b38298120d9d92603fa4e1e31d489887"
)
PHONE_TOKEN_ROWS_PER_LAYER_MIN = 6_800
PHONE_TOKEN_ROWS_PER_LAYER_MAX = 6_925


class PlanError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_bridge_allocator(path: Path, allocator: str) -> None:
    result = subprocess.run(
        [str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    usage = result.stdout + "\n" + result.stderr
    require(
        result.returncode == 2 and allocator in usage,
        f"bridge allocator capability: {allocator}",
    )


def require_phone_worker_capabilities(usage: str) -> None:
    required = {
        "--alternate-columns",
        "--column-quantum",
        "--max-tokens",
        "--staged-dmabuf",
    }
    require(
        all(value in usage for value in required),
        "phone worker split protocol capabilities",
    )


def require_phone_session_capabilities(source: str) -> None:
    required = {
        "S41_FFN_ALTERNATE_COLUMNS",
        "S41_FFN_COLUMN_QUANTUM",
        "S41_FFN_MAX_TOKENS",
        "S41_FFN_STAGED_DMABUF",
    }
    require(
        all(value in source for value in required),
        "phone session split protocol capabilities",
    )


def read_trace(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    require(len(rows) == 74, "source trace request count")
    require(digest_file(path) == TRACE_SHA256, "source trace hash")
    cold = [row for row in rows if row.get("model_id") == SOURCE_COLD_MODEL]
    require(
        len(cold) == 17
        and sum(row.get("input_tokens", 0) for row in cold) == 11_476
        and sum(row.get("output_tokens", 0) for row in cold) == 6_919
        and all(
            row.get("schema") == "s41-gemma-qwen-request-semantic-source-v1"
            and row.get("input_tokens") == row.get("source_input_tokens")
            and row.get("output_tokens") == row.get("source_output_tokens")
            and len(row.get("prompt_tokens", ())) == row.get("input_tokens")
            for row in cold
        ),
        "cold cohort geometry",
    )
    cold_bytes = b"".join(
        (
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        for row in cold
    )
    require(
        hashlib.sha256(cold_bytes).hexdigest() == COLD_WORK_SHA256,
        "cold cohort hash",
    )
    return rows, cold


def placement_candidates() -> tuple[LayerPlacementCandidate, ...]:
    return (
        LayerPlacementCandidate(
            candidate_id="gemma-f16-cuda-full-v1",
            model_id=TARGET_MODEL_ID,
            model_sha256=MODEL_SHA256,
            total_layers=48,
            cpu_prefix_layers=0,
            gpu_suffix_layers=48,
            runtime_gpu_layers=49,
            gpu_weight_bytes=MODEL_BYTES,
            gpu_kv_bytes=3_388_997_632,
            gpu_compute_bytes=290_455_552,
            cpu_weight_bytes=0,
            cpu_kv_bytes=0,
            cpu_compute_bytes=0,
            latency_us=None,
            latency_sample_count=0,
            status="estimated",
            placement_verified=False,
            evidence_ids=("oversized-model-file-v1",),
        ),
        LayerPlacementCandidate(
            candidate_id="gemma-f16-cpu23-cuda25-v1",
            model_id=TARGET_MODEL_ID,
            model_sha256=MODEL_SHA256,
            total_layers=48,
            cpu_prefix_layers=23,
            gpu_suffix_layers=25,
            runtime_gpu_layers=25,
            gpu_weight_bytes=12_914_765_332,
            gpu_kv_bytes=1_694_498_816,
            gpu_compute_bytes=290_455_552,
            cpu_weight_bytes=12_914_744_361,
            cpu_kv_bytes=1_694_498_816,
            cpu_compute_bytes=0,
            latency_us=779_833_719,
            latency_sample_count=3,
            status="measured",
            placement_verified=True,
            evidence_ids=(
                "sha256:" + CONTROL_RESULT_SHA256,
                "fp16-control-r1",
                "fp16-control-r2",
            ),
        ),
    )


def scheduling_candidates(cold: list[dict[str, Any]]) -> CandidateSet:
    members = tuple(row["event_id"] for row in cold)
    model = ModelIdentity(
        model_id=TARGET_MODEL_ID,
        model_hash="sha256:" + MODEL_SHA256,
        architecture="gemma4",
        weight_format="f16-storage-proxy",
        weight_bytes=MODEL_BYTES,
    )
    unit = SchedulingUnit(
        unit_id=UNIT_ID,
        kind=UnitKind.COHORT,
        member_request_ids=members,
        work_set_hash=make_work_set_hash(members),
        workload_id=WORKLOAD_ID,
        model=model,
        arrival_us=0,
        deadline_us=3_600_000_000,
        input_tokens=11_476,
        output_tokens=6_919,
        features={
            "runtime_gpu_layers": 25,
            "source_trace_requests": 74,
        },
        quality_requirement=QualityClass.SEMANTIC,
        semantics={"ignore_eos": True, "sampling": "temperature-zero"},
    )
    applicability = ApplicabilityContract.exact_for(unit)
    quality = QualityContract(
        quality_class=QualityClass.SEMANTIC,
        validation_id="fp16-storage-proxy-performance-placement-v1",
    )
    gpu = ResourceRequirement("cuda0", "gpu", "layer-suffix", 1)
    cpu = ResourceRequirement("cpu-package-0", "cpu", "layer-prefix", 1)
    phone = ResourceRequirement("op15-htp", "phone-npu", "ffn-slice", 1)
    usb = ResourceRequirement("op15-usb", "usb", "activation-transfer", 1)
    selected = placement_candidates()[1]
    base = dict(
        workload_id=WORKLOAD_ID,
        applicability=applicability,
        energy=(),
        quality=quality,
        phase_leases=(),
        placement_verified=True,
        source_profile_id=PROFILE_ID,
    )
    control = RouteAlternative(
        route_id=CONTROL_ROUTE,
        baseline=True,
        placement_granularity=PlacementGranularity.LAYER,
        maturity=RouteMaturity.MEASURED,
        latency_us=MetricEstimate(
            mean=779_833_719,
            upper=789_753_448,
            lower=768_271_901,
            sample_count=3,
            measured=True,
        ),
        overlap=OverlapEstimate("not_applicable", None, None, 0),
        resources=(gpu, cpu),
        memory=(
            MemoryRequirement("cuda0", "cuda0-vram", selected.gpu_total_bytes, True),
            MemoryRequirement("cpu-package-0", "host-ram", selected.cpu_total_bytes, True),
        ),
        residency=(),
        evidence_ids=("sha256:" + CONTROL_RESULT_SHA256,),
        **base,
    )
    treatment = RouteAlternative(
        route_id=OFFLOAD_ROUTE,
        baseline=False,
        placement_granularity=PlacementGranularity.OPERATOR,
        maturity=RouteMaturity.MEASURED,
        latency_us=MetricEstimate(
            mean=712_861_046,
            upper=712_861_046,
            lower=712_861_046,
            sample_count=1,
            measured=True,
        ),
        overlap=OverlapEstimate(
            status="measured",
            exposed_join_wait_ppm=123_827,
            upper_join_wait_ppm=250_000,
            sample_count=27_485,
        ),
        resources=(gpu, cpu, phone, usb),
        memory=(
            MemoryRequirement("cuda0", "cuda0-vram", selected.gpu_total_bytes, True),
            MemoryRequirement("cpu-package-0", "host-ram", selected.cpu_total_bytes, True),
            MemoryRequirement("op15-htp", "op15-ram", 3_255_880_909, True),
        ),
        residency=(),
        evidence_ids=("sha256:" + OFFLOAD_RESULT_SHA256,),
        **base,
    )
    return CandidateSet(PROFILE_ID, unit, (control, treatment))


def artifact_rows(
    paths: Mapping[str, str], hashes: Mapping[str, str]
) -> tuple[BackendArtifact, ...]:
    require(set(paths) == set(hashes), "artifact path and hash roles")
    return tuple(
        BackendArtifact(role, paths[role], hashes[role]) for role in sorted(paths)
    )


def build_plan(
    *,
    trace_path: Path,
    mode: str,
    artifact_paths: Mapping[str, str],
    artifact_hashes: Mapping[str, str],
    lib_dir: str,
    adb_port: int,
    phone_serial: str,
    split_policy_variant: str = "qualified",
    shape_calibration: Path = DEFAULT_CALIBRATION,
    gpu_occupied_bytes: int = 0,
    cpu_occupied_bytes: int = 0,
):
    require(mode in {"control", "shadow"}, "scheduler mode")
    require(
        split_policy_variant in SPLIT_POLICY_VARIANTS,
        "split policy variant",
    )
    require(
        mode == "shadow" or split_policy_variant == "qualified",
        "control cannot carry an unqualified split policy",
    )
    _, cold = read_trace(trace_path)
    candidates = scheduling_candidates(cold)
    decision = CohortPlanner(CohortPolicy(
        minimum_samples=1,
        max_exposed_join_wait_ppm=250_000,
    )).schedule(candidates, mode)
    expected_route = CONTROL_ROUTE if mode == "control" else OFFLOAD_ROUTE
    require(decision.route_id == expected_route, "cohort route decision")
    execution_mode = "cuda-cpu" if mode == "control" else "cuda-cpu-op15"

    gpu_capacity = DeviceMemoryCapacity(
        resource_id="cuda0-vram",
        capacity_bytes=17_175_674_880,
        occupied_bytes=gpu_occupied_bytes,
        reserve_bytes=536_870_912,
    )
    cpu_capacity = DeviceMemoryCapacity(
        resource_id="host-ram",
        capacity_bytes=32_862_289_920,
        occupied_bytes=cpu_occupied_bytes,
        reserve_bytes=2_147_483_648,
    )
    placement = CapacityPlanner(minimum_samples=3).plan(
        route_id=decision.route_id,
        candidates=placement_candidates(),
        gpu_capacity=gpu_capacity,
        cpu_capacity=cpu_capacity,
    )

    artifacts = artifact_rows(artifact_paths, artifact_hashes)
    by_role = {row.role: row for row in artifacts}
    offload = None
    split_balance = None
    calibration_hash = None
    if mode == "shadow":
        split, split_balance, calibration_hash = materialize_gemma_policy(
            split_policy_variant, shape_calibration
        )
        require(
            split.layer_ids == placement.cpu_layer_ids,
            "split and placement layers differ",
        )
        transport = PhoneTransportContract(
            transport_id="functionfs-dmabuf-f16-op15-v1",
            protocol="s41-ffn-split-flex-v2",
            host_endpoint="libusb-bulk",
            phone_endpoint="functionfs-dmabuf",
            allocator="malloc-split",
            io_type="f16",
            payload_offset_bytes=128,
            max_payload_bytes=split.n_embd * split.max_tokens * split.element_bytes,
            queue_depth=1,
            usb_speed_mbps=5000,
            usb_vendor_product="18d1:2d00",
            phone_resource_id="op15-htp",
            transport_resource_id="op15-usb",
            usb_root_resource_id="usb-root-0",
            bridge_residency_id="bridge:" + artifact_hashes["bridge"],
            worker_residency_id="worker:" + artifact_hashes["phone_worker"],
            reset_generation=0,
            max_reset_recoveries=0,
        )
        backend_roles = {
            "bridge",
            "cold_model",
            "phone_model",
            "phone_session",
            "phone_worker",
            "restore_usb",
        }
        backend = PhoneBackendConfig(
            backend_id="op15-htp-functionfs-dmabuf-f16-v1",
            phone_serial=phone_serial,
            adb_port=adb_port,
            compute_backend="HTP0",
            layer_spec=placement.cpu_layer_spec,
            bridge_bind="127.0.0.1",
            bridge_port=25660,
            server_timeout_ms=35_000,
            session_timeout_s=1800,
            max_requests=60_000,
            artifacts=tuple(by_role[role] for role in sorted(backend_roles)),
            resident_weight_bytes=3_255_880_909,
            resident_weight_budget_bytes=3_463_438_336,
        )
        offload = OperatorOffloadContract(
            route_id=decision.route_id,
            host_resource_id="cpu-package-0",
            split=split,
            transport=transport,
            backend=backend,
        )

    phone_layers = 23 if offload is not None else 0
    phone_token_rows_min = PHONE_TOKEN_ROWS_PER_LAYER_MIN * phone_layers
    phone_token_rows_max = PHONE_TOKEN_ROWS_PER_LAYER_MAX * phone_layers
    phone_transfer_bytes_min = phone_token_rows_min * 3840 * 2
    phone_transfer_bytes_max = phone_token_rows_max * 3840 * 2
    positive_phone_columns = (
        [
            row.phone_columns
            for row in offload.split.buckets
            if row.phone_columns > 0
        ]
        if offload is not None else [0]
    )
    phone_columns_min = min(positive_phone_columns)
    phone_columns_max = max(positive_phone_columns)
    expected_work = {
        "cold_input_tokens": 11_476,
        "cold_output_tokens": 6_919,
        "cpu_prefix_layers": 23,
        "gpu_suffix_layers": 25,
        "phone_calls_max": phone_token_rows_max,
        "phone_calls_min": phone_layers,
        "phone_layer_count": phone_layers,
        "phone_macs_max": (
            phone_token_rows_max * 3 * 3840 * phone_columns_max
        ),
        "phone_macs_min": (
            phone_token_rows_min * 3 * 3840 * phone_columns_min
        ),
        "phone_token_rows_per_layer_max": (
            PHONE_TOKEN_ROWS_PER_LAYER_MAX if offload is not None else 0
        ),
        "phone_token_rows_per_layer_min": (
            PHONE_TOKEN_ROWS_PER_LAYER_MIN if offload is not None else 0
        ),
        "phone_transfer_bytes_max": phone_transfer_bytes_max,
        "phone_transfer_bytes_min": phone_transfer_bytes_min,
    }
    runtime_bindings: dict[str, bool | int | str] = {
        "arrival_mode": "backlog",
        "batch_size": 4096,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "context": 32768,
        "cold_work_sha256": "sha256:" + COLD_WORK_SHA256,
        "dispatch_order": "source",
        "gpu_preflight_occupied_bytes": gpu_occupied_bytes,
        "host_preflight_occupied_bytes": cpu_occupied_bytes,
        "kv_offload": True,
        "lib_dir": lib_dir,
        "mode": execution_mode,
        "n_gpu_layers": placement.selected.runtime_gpu_layers,
        "parallel": 8,
        "port": 18482,
        "source_trace_requests": 74,
        "trace_filter": "cold",
        "trace_path": str(trace_path),
        "ubatch_size": 512,
    }
    epoch_key = canonical_sha256({
        "artifacts": artifact_hashes,
        "cold_work_sha256": COLD_WORK_SHA256,
        "gemma_split_calibration_sha256": calibration_hash,
        "gemma_split_policy": (
            None if offload is None else offload.split.to_json()
        ),
        "gpu_capacity": gpu_capacity.to_json(),
        "model_sha256": MODEL_SHA256,
        "profile_id": PROFILE_ID,
        "trace_sha256": TRACE_SHA256,
    })
    return build_execution_plan(
        plan_id=(
            f"burstgpt-cold17-gpu-overflow-{mode}-"
            f"{split_policy_variant}-v1"
        ),
        epoch_key=epoch_key,
        decision=decision,
        execution_mode=execution_mode,
        trace_sha256=TRACE_SHA256,
        request_count=17,
        input_tokens=11_476,
        output_tokens=6_919,
        model_hashes={"cold": MODEL_SHA256},
        artifacts=artifacts,
        runtime_bindings=runtime_bindings,
        expected_work=expected_work,
        offload=offload,
        layer_placement=placement,
        evidence_ids=tuple(dict.fromkeys((
            "sha256:" + CONTROL_RESULT_SHA256,
            "sha256:" + OFFLOAD_RESULT_SHA256,
            "sha256:" + TRACE_SHA256,
            *(() if split_balance is None else split_balance.evidence_ids),
            *(
                ()
                if calibration_hash is None
                else ("sha256:" + calibration_hash,)
            ),
        ))),
        admission_phase=(
            "shadow_shape_balance_energy_calibration"
            if split_policy_variant == "shape-balanced"
            else "shadow_acquisition_energy_not_yet_promoted"
        ),
    )


def adb_sha256(adb_port: int, serial: str, path: str) -> str:
    result = subprocess.run(
        ["adb", "-P", str(adb_port), "-s", serial, "shell", "sha256sum", path],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    fields = result.stdout.split()
    require(len(fields) >= 2 and fields[1] == path, f"phone artifact hash: {path}")
    return fields[0]


def adb_text(adb_port: int, serial: str, path: str) -> str:
    result = subprocess.run(
        ["adb", "-P", str(adb_port), "-s", serial, "shell", "cat", path],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout


def adb_worker_usage(adb_port: int, serial: str, path: str) -> str:
    root = path.rsplit("/", 1)[0]
    require("'" not in root and "'" not in path, "phone worker path quoting")
    result = subprocess.run(
        [
            "adb", "-P", str(adb_port), "-s", serial, "shell",
            f"su -c 'LD_LIBRARY_PATH={root} {path} 2>&1'",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    require(result.returncode == 2, "phone worker capability probe")
    return result.stdout + result.stderr


def gpu_occupied_bytes() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    fields = [item.strip() for item in result.stdout.strip().split(",")]
    require(len(fields) == 2, "GPU memory fields")
    total = int(fields[0]) * 1024 * 1024
    used = int(fields[1]) * 1024 * 1024
    require(total == 17_175_674_880 and 0 <= used < total, "GPU memory identity")
    return used


def host_occupied_bytes() -> int:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, _, raw = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            fields = raw.split()
            require(len(fields) == 2 and fields[1] == "kB", "host memory fields")
            values[key] = int(fields[0]) * 1024
    require(
        values.get("MemTotal") == 32_862_289_920
        and 0 < values.get("MemAvailable", 0) <= values["MemTotal"],
        "host memory identity",
    )
    return values["MemTotal"] - values["MemAvailable"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "shadow"), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lib-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--adb-port", type=int, required=True)
    parser.add_argument("--phone-serial", required=True)
    parser.add_argument("--phone-model")
    parser.add_argument("--phone-session")
    parser.add_argument("--phone-worker")
    parser.add_argument("--restore-usb")
    parser.add_argument(
        "--split-policy-variant",
        choices=SPLIT_POLICY_VARIANTS,
        default="qualified",
    )
    parser.add_argument(
        "--shape-calibration", type=Path, default=DEFAULT_CALIBRATION
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(args.server.is_file(), "server artifact")
        require(args.model.is_file() and args.model.stat().st_size == MODEL_BYTES, "model artifact")
        require(args.lib_dir.is_dir(), "runtime library directory")
        model_hash = digest_file(args.model)
        require(model_hash == MODEL_SHA256, "model artifact hash")
        paths = {
            "cold_model": str(args.model),
            "cold_server": str(args.server),
        }
        hashes = {
            "cold_model": model_hash,
            "cold_server": digest_file(args.server),
        }
        if args.mode == "shadow":
            require(
                args.bridge is not None and args.bridge.is_file()
                and all((args.phone_model, args.phone_session, args.phone_worker, args.restore_usb)),
                "treatment artifacts",
            )
            require_bridge_allocator(args.bridge, "malloc-split")
            require_phone_worker_capabilities(adb_worker_usage(
                args.adb_port, args.phone_serial, args.phone_worker
            ))
            require_phone_session_capabilities(adb_text(
                args.adb_port, args.phone_serial, args.phone_session
            ))
            paths.update({
                "bridge": str(args.bridge),
                "phone_model": args.phone_model,
                "phone_session": args.phone_session,
                "phone_worker": args.phone_worker,
                "restore_usb": args.restore_usb,
            })
            hashes.update({
                "bridge": digest_file(args.bridge),
                "phone_model": adb_sha256(args.adb_port, args.phone_serial, args.phone_model),
                "phone_session": adb_sha256(args.adb_port, args.phone_serial, args.phone_session),
                "phone_worker": adb_sha256(args.adb_port, args.phone_serial, args.phone_worker),
                "restore_usb": adb_sha256(args.adb_port, args.phone_serial, args.restore_usb),
            })
            require(hashes["phone_model"] == MODEL_SHA256, "phone model hash")
        plan = build_plan(
            trace_path=args.trace,
            mode=args.mode,
            artifact_paths=paths,
            artifact_hashes=hashes,
            lib_dir=str(args.lib_dir),
            adb_port=args.adb_port,
            phone_serial=args.phone_serial,
            split_policy_variant=args.split_policy_variant,
            shape_calibration=args.shape_calibration,
            gpu_occupied_bytes=gpu_occupied_bytes(),
            cpu_occupied_bytes=host_occupied_bytes(),
        )
        write_execution_plan(args.output, plan)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"GPU overflow planning failed: {exc}\n")
    print(json.dumps({
        "cpu_layers": plan.layer_placement.cpu_layer_spec,
        "gpu_layers": plan.layer_placement.gpu_layer_spec,
        "mode": plan.execution_mode,
        "plan_sha256": plan.plan_sha256,
        "reason": plan.decision.reason,
        "route_id": plan.decision.route_id,
        "split_policy": (
            None if plan.offload is None else plan.offload.split.table
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
