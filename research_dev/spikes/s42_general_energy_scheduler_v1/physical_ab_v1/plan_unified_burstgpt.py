#!/usr/bin/env python3
"""Build a unified-scheduler plan for the certified BurstGPT cohort."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    BackendArtifact,
    CohortPolicy,
    CohortPlanner,
    OperatorOffloadContract,
    OperatorSplitPolicy,
    PhoneBackendConfig,
    PhoneTransportContract,
    QualityClass,
    build_execution_plan,
    load_certified_cohort,
    write_execution_plan,
)


WORKLOAD_ID = "burstgpt-source-two-model-v1"
UNIT_ID = "burstgpt-source-74-v1"
BOUNDARY_ID = "cpu-package+gpu-board+whole-phone-paid-interval-v3"
HOT_SOURCE_MODEL = "gemma-4-12b-it-q8_0"
COLD_SOURCE_MODEL = "qwen3-14b-q4_k_m"
CONTROL_ROUTE = "i3-cold-cpu-control-v1"
OFFLOAD_ROUTE = "i3-cold-cpu-op15-ffn-v1"


class PlanError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object: {path}")
    return value


def read_trace(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    require(len(rows) == 74, "trace request count")
    require(
        Counter(row.get("model_id") for row in rows)
        == Counter({HOT_SOURCE_MODEL: 57, COLD_SOURCE_MODEL: 17}),
        "trace model-role counts",
    )
    require(
        all(
            row.get("schema") == "s41-gemma-qwen-request-semantic-source-v1"
            and row.get("input_tokens") == row.get("source_input_tokens")
            and row.get("output_tokens") == row.get("source_output_tokens")
            and len(row.get("prompt_tokens", [])) == row.get("input_tokens")
            for row in rows
        ),
        "trace token geometry",
    )
    require(len({row.get("event_id") for row in rows}) == len(rows), "trace ids")
    return rows


def artifact_rows(
    paths: Mapping[str, str], hashes: Mapping[str, str]
) -> tuple[BackendArtifact, ...]:
    require(set(paths) == set(hashes), "artifact path/hash roles")
    return tuple(
        BackendArtifact(role, paths[role], hashes[role])
        for role in sorted(paths)
    )


def build_plan(
    *,
    trace_path: Path,
    compiled_path: Path,
    epoch_path: Path,
    contracts_path: Path,
    mode: str,
    artifact_paths: Mapping[str, str],
    artifact_hashes: Mapping[str, str],
    cuda_lib_dir: str,
    cold_lib_dir: str,
    adb_port: int,
    phone_serial: str,
):
    require(mode in {"control", "enforce"}, "scheduler mode")
    rows = read_trace(trace_path)
    epoch = read_object(epoch_path)
    certificate = read_object(compiled_path)
    trace_hash = digest_file(trace_path)
    bindings = epoch["bindings"]
    require(trace_hash == bindings["workload"]["sha256"], "trace hash")

    profile = load_certified_cohort(
        compiled_path=compiled_path,
        epoch_path=epoch_path,
        runtime_contracts_path=contracts_path,
        workload_id=WORKLOAD_ID,
        unit_id=UNIT_ID,
        member_request_ids=[row["event_id"] for row in rows],
        quality_requirement=QualityClass.SEMANTIC,
        boundary_id=BOUNDARY_ID,
    )
    decision = CohortPlanner(CohortPolicy()).schedule(
        profile.candidates, mode
    )
    require(
        decision.route_id
        == (CONTROL_ROUTE if mode == "control" else OFFLOAD_ROUTE),
        "scheduler route decision",
    )
    dispatch = dict(profile.dispatch_contracts[decision.route_id])
    execution_mode = {
        "cpu-control": "cpu",
        "cpu-htp-operator-split": "op15",
    }.get(dispatch.get("route_mode"))
    require(execution_mode is not None, "dispatch route mode")

    artifacts = artifact_rows(artifact_paths, artifact_hashes)
    artifacts_by_role = {item.role: item for item in artifacts}
    offload = None
    if execution_mode == "op15":
        policy_binding = bindings["policy"]
        split = OperatorSplitPolicy.from_table(
            policy_id=dispatch["policy_id"],
            operator_family=dispatch["operator_family"],
            layer_ids=range(bindings["models"]["cold"]["n_layer"]),
            n_embd=bindings["models"]["cold"]["n_embd"],
            eligible_columns=policy_binding["max_columns"],
            max_tokens=bindings["concurrency"]["cold_ubatch_size"],
            column_quantum=policy_binding["column_quantum"],
            alternate_columns=(9664,),
            io_type=policy_binding["io"],
            weight_layout=policy_binding["weight_layout"],
            table=dispatch["split_table"],
        )
        transport = PhoneTransportContract(
            transport_id="functionfs-dmabuf-f16-op15-v1",
            protocol=bindings["transport"]["protocol"],
            host_endpoint=bindings["transport"]["desktop_endpoint"],
            phone_endpoint=bindings["transport"]["phone_endpoint"],
            allocator=bindings["transport"]["allocator"],
            io_type=bindings["transport"]["io_type"],
            payload_offset_bytes=128,
            max_payload_bytes=(
                split.n_embd * split.max_tokens * split.element_bytes
            ),
            queue_depth=1,
            usb_speed_mbps=5000,
            usb_vendor_product="18d1:2d00",
            phone_resource_id="op15-htp",
            transport_resource_id="op15-usb",
            usb_root_resource_id="usb-root-0",
            bridge_residency_id=(
                "bridge:" + artifact_hashes["bridge"].removeprefix("sha256:")
            ),
            worker_residency_id=(
                "worker:" + artifact_hashes["phone_worker"].removeprefix("sha256:")
            ),
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
            backend_id="op15-htp-functionfs-dmabuf-v1",
            phone_serial=phone_serial,
            adb_port=adb_port,
            compute_backend="HTP0",
            layer_spec="0-47",
            bridge_bind="127.0.0.1",
            bridge_port=25660,
            server_timeout_ms=35000,
            session_timeout_s=1800,
            max_requests=120000,
            artifacts=tuple(
                artifacts_by_role[role] for role in sorted(backend_roles)
            ),
        )
        offload = OperatorOffloadContract(
            route_id=decision.route_id,
            host_resource_id="cpu-cold",
            split=split,
            transport=transport,
            backend=backend,
        )

    workload = bindings["workload"]
    phone_work = read_object(
        compiled_path.parent / "CERTIFICATES_4060TI_OP15_V1.json"
    )["campaign"]["phone_work"]
    expected_work = {
        "phone_calls": phone_work["calls"] if offload is not None else 0,
        "phone_download_bytes": (
            phone_work["download_bytes"] if offload is not None else 0
        ),
        "phone_macs": phone_work["phone_macs"] if offload is not None else 0,
        "phone_upload_bytes": (
            phone_work["upload_bytes"] if offload is not None else 0
        ),
    }
    policy_binding = bindings["policy"]
    cold_runtime = bindings["runtimes"]["cold"]
    hot_runtime = bindings["runtimes"]["hot"]
    runtime_bindings: dict[str, bool | int | str] = {
        "cold_batch_size": cold_runtime["batch_size"],
        "cold_context": cold_runtime["context"],
        "cold_lib_dir": cold_lib_dir,
        "cold_parallel": cold_runtime["parallel"],
        "cold_repack": cold_runtime["repack"],
        "cold_threads": cold_runtime["threads"],
        "cold_ubatch_size": cold_runtime["ubatch_size"],
        "hot_context": hot_runtime["context"],
        "cuda_lib_dir": cuda_lib_dir,
        "hot_parallel": hot_runtime["parallel"],
        "mode": execution_mode,
        "phone_dense_ffn_split": bool(dispatch["phone_dense_ffn_split"]),
        "request_workers": workload["requests"],
        "split_io": policy_binding["io"],
        "split_max_columns": policy_binding["max_columns"],
        "split_policy_id": policy_binding["id"],
        "split_table": policy_binding["table"],
        "trace_path": str(trace_path),
    }
    evidence = certificate["physical_evidence"]
    return build_execution_plan(
        plan_id=f"burstgpt-source-74-{mode}-v1",
        epoch_key=profile.epoch_key,
        decision=decision,
        execution_mode=execution_mode,
        trace_sha256=trace_hash,
        request_count=workload["requests"],
        input_tokens=workload["input_tokens"],
        output_tokens=workload["output_tokens"],
        model_hashes={
            "cold": bindings["models"]["cold"]["file_sha256"],
            "hot": bindings["models"]["hot"]["file_sha256"],
        },
        artifacts=artifacts,
        runtime_bindings=runtime_bindings,
        expected_work=expected_work,
        offload=offload,
        evidence_ids=(
            evidence["aggregate_file_sha256"],
            evidence["aggregate_record_sha256"],
            evidence["trace_sha256"],
        ),
    )


def adb_sha256(adb_port: int, serial: str, path: str) -> str:
    result = subprocess.run(
        ["adb", "-P", str(adb_port), "-s", serial, "shell", "sha256sum", path],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    fields = result.stdout.split()
    require(len(fields) >= 2 and fields[1] == path, f"phone artifact hash: {path}")
    return fields[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--epoch", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--mode", choices=("control", "enforce"), required=True)
    parser.add_argument("--adb-port", type=int, required=True)
    parser.add_argument("--phone-serial", required=True)
    parser.add_argument("--hot-server", type=Path, required=True)
    parser.add_argument("--cold-server", type=Path, required=True)
    parser.add_argument("--hot-model", type=Path, required=True)
    parser.add_argument("--cold-model", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--cold-lib-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--phone-session", required=True)
    parser.add_argument("--phone-worker", required=True)
    parser.add_argument("--phone-model", required=True)
    parser.add_argument("--restore-usb", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(args.cuda_lib_dir.is_dir(), "CUDA library directory")
        require(args.cold_lib_dir.is_dir(), "cold library directory")
        epoch = read_object(args.epoch)
        bindings = epoch["bindings"]
        cold_manifest = {
            row["name"]: row["sha256"]
            for row in bindings["runtimes"]["cold"]["manifest"]
        }
        hot_manifest = {
            row["name"]: row["sha256"]
            for row in bindings["runtimes"]["hot"]["manifest"]
        }
        expected = {
            "bridge": bindings["artifacts"]["desktop_bridge"]["sha256"],
            "cold_model": bindings["models"]["cold"]["file_sha256"],
            "cold_server": cold_manifest["llama-server"],
            "hot_model": bindings["models"]["hot"]["file_sha256"],
            "hot_server": hot_manifest["llama-server"],
            "phone_model": bindings["models"]["cold"]["file_sha256"],
            "phone_worker": bindings["artifacts"]["phone_worker"]["sha256"],
        }
        paths = {
            "bridge": str(args.bridge),
            "cold_model": str(args.cold_model),
            "cold_server": str(args.cold_server),
            "hot_model": str(args.hot_model),
            "hot_server": str(args.hot_server),
            "phone_model": args.phone_model,
            "phone_session": args.phone_session,
            "phone_worker": args.phone_worker,
            "restore_usb": args.restore_usb,
        }
        hashes = {
            "bridge": digest_file(args.bridge),
            "cold_model": digest_file(args.cold_model),
            "cold_server": digest_file(args.cold_server),
            "hot_model": digest_file(args.hot_model),
            "hot_server": digest_file(args.hot_server),
            "phone_model": expected["phone_model"],
            "phone_session": adb_sha256(
                args.adb_port, args.phone_serial, args.phone_session
            ),
            "phone_worker": adb_sha256(
                args.adb_port, args.phone_serial, args.phone_worker
            ),
            "restore_usb": adb_sha256(
                args.adb_port, args.phone_serial, args.restore_usb
            ),
        }
        require(
            all(hashes[role] == digest for role, digest in expected.items()),
            "certified artifact hash",
        )
        plan = build_plan(
            trace_path=args.trace,
            compiled_path=args.compiled,
            epoch_path=args.epoch,
            contracts_path=args.contracts,
            mode=args.mode,
            artifact_paths=paths,
            artifact_hashes=hashes,
            cuda_lib_dir=str(args.cuda_lib_dir),
            cold_lib_dir=str(args.cold_lib_dir),
            adb_port=args.adb_port,
            phone_serial=args.phone_serial,
        )
        write_execution_plan(args.output, plan)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"unified BurstGPT planning failed: {exc}\n")
    print(json.dumps({
        "mode": plan.execution_mode,
        "output": str(args.output),
        "plan_sha256": plan.plan_sha256,
        "reason": plan.decision.reason,
        "route_id": plan.decision.route_id,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
