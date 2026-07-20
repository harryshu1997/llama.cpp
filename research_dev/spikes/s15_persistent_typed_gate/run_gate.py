#!/usr/bin/env python3
"""Run the typed persistent executor through a real OP15 head and CUDA tail."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPIKES = HERE.parent
ROOT = HERE.parents[2]
PERSISTENT_LIVE = SPIKES / "s15_persistent_live_launcher"
ONE_SHOT_LIVE = SPIKES / "s15_live_launcher"
PERSISTENT = SPIKES / "s15_persistent_runtime"
RUNTIME = SPIKES / "s15_runtime_dispatch"
S14 = SPIKES / "s14_mixed_streaming_scheduler"
DIRECT = SPIKES / "s15_persistent_host_tail"
ARRIVAL = SPIKES / "s15_arrival_faithful_b32"
PERSISTENT_B32 = SPIKES / "s15_persistent_b32"
for path in (
        PERSISTENT_LIVE, ONE_SHOT_LIVE, PERSISTENT, RUNTIME, S14,
        DIRECT, ARRIVAL, PERSISTENT_B32):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from executor_contract import BOUNDARY_SCHEMA, ExecutionRequest  # noqa: E402
from live_adapter import PersistentLiveExecutor, PersistentLiveSessionTransport  # noqa: E402
from live_contract import (  # noqa: E402
    FrozenRoute,
    LaunchSpec,
    canonical,
    digest,
    parse_host_placement,
    strict_line,
)
from persistent_bridge import CONFIG_SCHEMA as BRIDGE_CONFIG_SCHEMA  # noqa: E402
from persistent_transport import PersistentPreparedTransport  # noqa: E402
from physical_executor import PhysicalRouteBinding  # noqa: E402
from session_adapter import (  # noqa: E402
    PersistentWorkerCapability,
    StageNetSessionAdapter,
    StageNetSessionBinding,
    parse_session_cert,
)
import physical_launcher as base  # noqa: E402
import run_gate as prior  # noqa: E402
import run_recertification as recert  # noqa: E402


SCHEMA = "s15-persistent-typed-physical-gate-v1"
VERDICT = "TYPED_PERSISTENT_OP15_B32_PHYSICAL_PASS_ENERGY_UNKNOWN"
REMOTE = "/data/local/tmp/ls-s15-persistent-typed"
PORT = 5964
BATCH = 32
N_GEN = 8
DEADLINE_US = 5_000_000
RESULTS = HERE / "results"
COHORT = SPIKES / "s15_burst_cohort/cohort.json"
INPUT_MANIFEST = SPIKES / "s15_burst_cohort/input_manifest.json"
PROFILE = SPIKES / "s15_batch32_gate/results/gate_report.json"
EVIDENCE = DIRECT / "results/report.json"


class GateError(RuntimeError):
    pass


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def strict_json(path: Path, label: str) -> dict:
    payload = path.read_bytes()

    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise GateError(f"duplicate key in {label}: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict or canonical(value) != payload:
        raise GateError(f"{label} is not one canonical object")
    return value


def load_inputs() -> tuple[tuple[str, ...], str, str, str]:
    for path in (COHORT, INPUT_MANIFEST, PROFILE, EVIDENCE):
        if not path.is_file():
            raise GateError(f"required frozen input is missing: {path}")
    cohort = strict_json(COHORT, "cohort")
    manifest = strict_json(INPUT_MANIFEST, "input manifest")
    requests = cohort.get("requests")
    payloads = manifest.get("request_payloads")
    prompt = manifest.get("prompt_text")
    if type(requests) is not list or len(requests) != BATCH \
            or type(payloads) is not list or len(payloads) != BATCH \
            or type(prompt) is not str or not prompt:
        raise GateError("frozen cohort or input manifest has the wrong shape")
    request_ids = tuple(item.get("event_id") for item in requests)
    payload_ids = tuple(item.get("event_id") for item in payloads)
    if any(type(value) is not str or not value for value in request_ids) \
            or len(set(request_ids)) != BATCH or request_ids != payload_ids:
        raise GateError("frozen cohort and payload manifest do not match")
    return request_ids, prompt, file_digest(COHORT), file_digest(INPUT_MANIFEST)


def deploy() -> tuple[str, str]:
    if not prior.ANDROID_BIN.is_file() or not prior.HOST_BIN.is_file():
        raise GateError("current host or Android binary is missing")
    base.adb_checked("shell", f"rm -rf {REMOTE} && cp -a {base.REMOTE} {REMOTE}")
    pushed = base.adb("push", str(prior.ANDROID_BIN), f"{REMOTE}/llama-layersplit", timeout=240)
    if pushed.returncode != 0:
        raise GateError("failed to deploy the current Android worker")
    base.adb_checked("shell", f"chmod 755 {REMOTE}/llama-layersplit")
    expected = file_digest(prior.ANDROID_BIN)
    observed = base.adb_checked(
        "shell", f"sha256sum {REMOTE}/llama-layersplit",
    ).decode("ascii").split()[0]
    if expected != "sha256:" + observed:
        raise GateError("deployed Android worker digest mismatch")
    shard = base.adb_checked(
        "shell", f"sha256sum {base.SHARD}", timeout=180,
    ).decode("ascii").split()[0]
    if shard != base.SHARD_SHA256:
        raise GateError("phone shard digest mismatch")
    boot_id = base.adb_checked(
        "shell", "cat /proc/sys/kernel/random/boot_id",
    ).decode("ascii").strip()
    return boot_id, expected


def commands() -> tuple[list[str], list[str], dict[str, str]]:
    phone_shell = (
        f"cd {REMOTE} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
        "GGML_HEXAGON_MBUF=4192 LLAMA_LAYER_END=8 LAYERSPLIT_PLACEMENT_CERT=1 "
        f"./llama-layersplit -m {base.SHARD} --devices HTP0 -ngl 99 --mode stagenet "
        f"--port {PORT} -n {N_GEN} --driver-batch {BATCH} "
        f"--driver-context {base.CONTEXT} --driver-max-prefill {base.MAX_PREFILL}"
    )
    phone = ["adb", "-s", base.SERIAL, "shell", phone_shell]
    host = [
        str(prior.HOST_BIN), "-m", str(base.FULL_MODEL), "-ngl", "99",
        "--mode", "pipedriver", "--host", "127.0.0.1", "--port", str(PORT),
        "-n", str(N_GEN), "--driver-batch", str(BATCH),
        "--driver-context", str(base.CONTEXT),
        "--driver-max-prefill", str(base.MAX_PREFILL), "--persistent-jsonl",
    ]
    environment = {
        "CUDA_VISIBLE_DEVICES": base.GPU_UUID,
        "LD_LIBRARY_PATH": str(recert.HOST_DIR),
        "LLAMA_LAYER_START": "8",
        "LAYERSPLIT_PLACEMENT_CERT": "1",
    }
    return phone, host, environment


def route_record(boot_id: str, worker_sha256: str) -> FrozenRoute:
    return FrozenRoute(
        "gemma-op15-head-0-8-cuda-tail-8-48-b32",
        file_digest(PROFILE),
        file_digest(EVIDENCE),
        f"op15:{base.SERIAL}",
        boot_id,
        worker_sha256,
        (0, 8),
        (8, 48),
        15, 1, 1, 1, BATCH, N_GEN,
    )


def route_json(route: FrozenRoute) -> dict:
    return {
        "route_id": route.route_id,
        "profile_id": route.profile_id,
        "evidence_sha256": route.evidence_sha256,
        "device_id": route.device_id,
        "device_boot_id": route.device_boot_id,
        "worker_binary_sha256": route.worker_binary_sha256,
        "layer_range": list(route.layer_range),
        "host_tail_range": list(route.host_tail_range),
        "route_epoch": route.route_epoch,
        "residency_epoch": route.residency_epoch,
        "device_boot_epoch": route.device_boot_epoch,
        "registry_generation": route.registry_generation,
        "batch_size": route.batch_size,
        "max_n_gen": route.max_n_gen,
    }


def request(route: FrozenRoute, spec: LaunchSpec, cohort_sha256: str,
            manifest_sha256: str) -> ExecutionRequest:
    return ExecutionRequest(
        spec.launch_id, route.route_id, route.profile_id, route.device_id,
        route.route_epoch, route.residency_epoch, spec.launch_id,
        route.device_boot_epoch, route.registry_generation,
        "gemma-4-12b|decode|head-0-8|b32",
        spec.request_ids, cohort_sha256, manifest_sha256,
        spec.deadline_us, BOUNDARY_SCHEMA,
    )


def finish_transport(outer: PersistentPreparedTransport) -> None:
    if outer.state == "READY":
        outer.drain()
    if outer.state not in ("FINALIZED",):
        outer.finalize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise GateError("physical execution requires --run")
    if RESULTS.exists():
        raise GateError("result directory already exists")
    RESULTS.mkdir(parents=True)
    outer = None
    thermal = None
    try:
        request_ids, prompt, cohort_sha256, manifest_sha256 = load_inputs()
        base.adb("shell", "pkill -9 llama-layersplit")
        base.adb("forward", "--remove", f"tcp:{PORT}")
        if base.adb("forward", f"tcp:{PORT}", f"tcp:{PORT}").returncode != 0:
            raise GateError("failed to install the OP15 port forward")
        boot_id, worker_sha256 = deploy()
        reference_dir = RESULTS / "reference"
        reference_dir.mkdir()
        reference = recert.reference_tokens(prompt, reference_dir)
        token_sha256 = digest(canonical({"token_ids": [reference] * BATCH}))
        phone_command, host_command, host_env = commands()
        mux_artifacts = RESULTS / "mux-artifacts"
        mux_config = {
            "schema": "s15-physical-mux-config-v1",
            "host_command": host_command,
            "phone_command": phone_command,
            "host_env": host_env,
            "artifact_root": str(mux_artifacts.resolve()),
            "host_layer_start": 8,
            "host_layer_end": 48,
            "host_backend": "CUDA0",
        }
        mux_config_path = RESULTS / "physical-mux.config.json"
        mux_config_path.write_bytes(canonical(mux_config))

        route = route_record(boot_id, worker_sha256)
        child_artifacts = RESULTS / "bridge-child-artifacts"
        bridge_config = {
            "schema": BRIDGE_CONFIG_SCHEMA,
            "route": route_json(route),
            "child_command": [
                sys.executable, str(HERE / "physical_mux.py"),
                "--config", str(mux_config_path.resolve()),
            ],
            "artifact_root": str(child_artifacts.resolve()),
            "child_ready_timeout_s": 600,
        }
        bridge_config_path = RESULTS / "bridge.config.json"
        bridge_config_path.write_bytes(canonical(bridge_config))

        specs = tuple(
            LaunchSpec(
                launch_id, prompt, N_GEN, request_ids,
                "STOP" if launch_id == 2 else "DETACH",
                DEADLINE_US, token_sha256,
            )
            for launch_id in (1, 2)
        )
        binding = PhysicalRouteBinding(
            route.route_id, route.profile_id, route.device_id, 1,
            worker_sha256, 1, 1, boot_id,
            cohort_sha256, manifest_sha256, route.layer_range, "HTP0", ("GET_ROWS",),
        )
        session = StageNetSessionAdapter(
            StageNetSessionBinding(worker_sha256, 1, boot_id, route.layer_range, 48),
            (PersistentWorkerCapability(worker_sha256, 2, True),),
        )
        outer = PersistentPreparedTransport(
            (
                sys.executable, str(PERSISTENT_LIVE / "persistent_bridge.py"),
                "--config", str(bridge_config_path.resolve()),
            ),
            RESULTS / "outer-artifacts", ready_timeout_s=620,
        )
        live = PersistentLiveSessionTransport(
            outer, route, binding, session,
            {value.launch_id: value for value in specs}, child_artifacts,
        )
        executor = PersistentLiveExecutor(binding, live)
        thermal_paths, thermal_start = base.discover_thermal_paths()
        thermal = base.ThermalMonitor(
            RESULTS / "thermal.log", RESULTS / "thermal.stderr.bin", thermal_paths,
        )
        thermal.wait_ready()

        executions = []
        now_us = 0
        for spec in specs:
            typed_request = request(route, spec, cohort_sha256, manifest_sha256)
            result = executor.launch(typed_request, now_us)
            result.validate(typed_request)
            if result.outcome != "completed" or len(result.certificates) != BATCH:
                raise GateError("typed physical launch did not complete every request")
            executions.append({
                "launch_id": spec.launch_id,
                "session_end": spec.session_end,
                "finish_us": result.finish_us,
                "certificate_count": len(result.certificates),
            })
            now_us = result.finish_us
        thermal_end = thermal.snapshot()
        thermal.stop()
        thermal = None
        finish_transport(outer)
        outer = None

        certs = []
        child_results = []
        host_placements = []
        for spec in specs:
            launch_dir = child_artifacts / f"launch-{spec.launch_id:06d}"
            certs.append(parse_session_cert((launch_dir / "session_cert.bin").read_bytes()))
            child = strict_line((launch_dir / "child_result.bin").read_bytes(), "child result")
            child_results.append(child)
            host_placements.append(parse_host_placement(
                (launch_dir / "host_placement.bin").read_bytes(),
                route.host_tail_range, child["host_pid"],
            ))
        if len({value["worker_pid"] for value in certs}) != 1 \
                or len({value["worker_boot_nonce"] for value in certs}) != 1 \
                or len({value["host_pid"] for value in child_results}) != 1:
            raise GateError("persistent process identity changed across exchanges")
        if [value["session_id"] for value in certs] != [1, 2] \
                or not certs[0]["reset_applied"] or certs[1]["reset_applied"]:
            raise GateError("persistent reset/session sequence is invalid")
        if not base.thermal_ok(thermal_end, base.THERMAL_END_MAX_MILLIC):
            raise GateError("phone thermal end gate failed")

        report = {
            "schema": SCHEMA,
            "verdict": VERDICT,
            "scope": "REAL_OP15_A6000_TYPED_TWO_EXCHANGES_SYNTHETIC_PAYLOAD_ENERGY_UNKNOWN",
            "route": route_json(route),
            "cohort_sha256": cohort_sha256,
            "input_manifest_sha256": manifest_sha256,
            "worker_binary_sha256": worker_sha256,
            "host_binary_sha256": file_digest(prior.HOST_BIN),
            "physical_mux_sha256": file_digest(HERE / "physical_mux.py"),
            "bridge_sha256": file_digest(PERSISTENT_LIVE / "persistent_bridge.py"),
            "live_adapter_sha256": file_digest(PERSISTENT_LIVE / "live_adapter.py"),
            "reference_tokens": reference,
            "executions": executions,
            "child_results": child_results,
            "phone_sessions": certs,
            "host_placements": host_placements,
            "thermal_start": thermal_start,
            "thermal_end": thermal_end,
            "phone_command": phone_command,
            "host_command": host_command,
            "formal_energy_claim": "NONE",
            "phone_energy": "UNKNOWN",
            "total_system_energy": "UNKNOWN",
            "problems": [],
        }
        (RESULTS / "report.json").write_bytes(canonical(report))
        print(json.dumps({
            "verdict": VERDICT,
            "host_pid": child_results[0]["host_pid"],
            "worker_pid": certs[0]["worker_pid"],
            "max_elapsed_us": max(value["elapsed_us"] for value in child_results),
        }, sort_keys=True))
        return 0
    finally:
        if outer is not None:
            outer.terminate()
            try:
                outer.finalize()
            except Exception:
                pass
        if thermal is not None:
            thermal.stop()
        base.adb("forward", "--remove", f"tcp:{PORT}")
        base.adb("shell", "pkill -9 llama-layersplit")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, base.LauncherError, prior.GateError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
