"""Bounded whole-phone memory measurement through canonical scheduler tickets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import threading
import time
import traceback

from . import runner
from .launch import write_new
from .offline_residency_gate import _SnapshotStore, _execution_json, _request_probe, _submit_request
from ...adapters import CanonicalOfflinePhoneResidencyPreloader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hash-cache-input", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=2)
    options = parser.parse_args()
    if not 1 <= options.repetitions <= 4:
        parser.error("repetitions must be between one and four")
    options.output.mkdir()
    for name in ("snapshots", "streams"):
        (options.output / name).mkdir()
    original = json.loads((options.base_run / "RUN_COMMAND_EXECUTION.json").read_text())
    args = runner._build_parser().parse_args(original[2:])
    args.output = options.output
    args.phone_session_root = "/data/local/tmp/" + options.output.name
    args.phone_whole_state_directory = args.phone_session_root + "-whole"
    args.phone_remote_hash_cache = options.output / "PHONE_HASH_CACHE.json"
    shutil.copyfile(options.hash_cache_input, args.phone_remote_hash_cache)
    models = runner._load_trace_models(args)
    scheduler, manifests, _ = runner._build_scheduler(args, models)
    aliases, merged, _, _ = runner._select_replay(args, models, manifests)
    rig = runner._build_rig(args, models, manifests, dict(args.transport_host_dependency))
    snapshots = _SnapshotStore(options.output / "snapshots", rig)
    observations, executions = [], []
    stop = threading.Event()
    observer = None
    started = False
    result = None
    try:
        write_new(options.output / "DIRECT_PREFLIGHT.json", rig.direct_phone_preflight().to_json())
        started = True
        rig.start(runner._warm_payload(models, aliases, options.output / "streams"))
        epoch_ns = time.monotonic_ns()
        rig.begin_offline_preload(epoch_ns)
        item = next(row for row in merged if row["model_id"] == models.expected_gemma.model_id)
        probe = _request_probe(item, aliases, options.output / "streams",
                               request_id="memory-control-preparation", arrival_us=1, seed=42)
        snapshot = snapshots.capture(probe.request, item["model_id"], 1, "initial")
        plan = snapshots.plan(scheduler, {item["model_id"]: (probe.request,)}, snapshot, epoch_ns)
        write_new(options.output / "OFFLINE_PLAN.json", plan.to_json())
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler, rig.backend(), epoch_ns=epoch_ns, snapshot_provider=snapshots.offline,
        )
        now = (time.monotonic_ns() - epoch_ns) // 1000
        stage = preloader.execute_next_stage(
            plan.plan_id, probe.payload, snapshot=snapshots.offline(plan.current_stage, now), observed_at_us=now,
        )
        write_new(options.output / "PREPARATION.json", stage.to_json())
        retained = dict(rig.direct_phone_residency_state)
        rig._android_phone_launcher.connect_ncm_control()
        for _ in range(40):
            _, health = rig.phone_runtime_observation()
            if health.get("valid"):
                break
            rig.request_runtime_observation_refresh()
            stop.wait(0.25)
        else:
            raise RuntimeError("Whole-phone memory probe lacks fresh device health")

        def observe() -> None:
            with (options.output / "MEMORY_SAMPLES.jsonl").open("x") as stream:
                while not stop.is_set():
                    with rig._lock:
                        states = tuple(state for state in rig._live_executors.values()
                                       if state.parameters.get("execution_adapter") == "android-llama-server-v1")
                    for state in states:
                        server = state.server
                        try:
                            value = rig._android_phone_launcher.probe_process_memory_peak(
                                server.process_identity, server._pid_file,
                            )
                            with rig._lock:
                                if rig._live_executors.get(state.executor_id) is not state:
                                    raise RuntimeError("Memory probe endpoint generation changed")
                            value.update(executor_id=state.executor_id, generation=state.generation,
                                         artifact_sha256=state.manifest.artifact_sha256)
                        except Exception as error:
                            value = {"error": repr(error), "sample_time_ns": time.monotonic_ns()}
                        observations.append(value)
                        stream.write(json.dumps(value, sort_keys=True) + "\n")
                        stream.flush()
                    stop.wait(0.5)

        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        item = next(row for row in merged if row["model_id"] == models.llama_model_id)
        for index in range(options.repetitions):
            request = _request_probe(
                item, aliases, options.output / "streams", request_id="whole-memory-" + str(index),
                arrival_us=(time.monotonic_ns() - epoch_ns) // 1000, seed=42,
            )
            coordinator, ticket = _submit_request(scheduler, rig, snapshots, epoch_ns, request, "calibration")
            print("SUBMITTED", index, ticket.decision.route_id, flush=True)
            try:
                completed = coordinator.drain(timeout_s=240)
            finally:
                coordinator.close()
            execution = completed.executions[request.request.request_id]
            value = _execution_json(execution)
            write_new(options.output / f"EXECUTION_{index}.json", value)
            executions.append(value)
            if (execution.command.adapter_parameters.get("execution_adapter") != "android-llama-server-v1"
                    or execution.recoveries):
                raise RuntimeError("Calibration did not execute wholly on phone; no route forced")
            print("COMPLETED", index, flush=True)
        if any(retained[key] != rig.direct_phone_residency_state[key]
               for key in ("phone_shards", "load_count_by_session")):
            raise RuntimeError("Whole-phone measurement changed retained HTP residency")
        stop.set()
        observer.join(timeout=5)
        if observer.is_alive():
            raise RuntimeError("Whole-phone memory observer did not stop")
        valid = [row for row in observations if "accounted_peak_bytes" in row]
        if (not valid or len(valid) != len(observations)
                or any(row["has_unbounded_accounting"] for row in valid)):
            raise RuntimeError("Memory peak accounting is incomplete; do not reduce reservation")
        result = {"status": "PASS", "scope": "tested request shape, not energy qualification",
                  "request_shape": item["row"], "retained_sessions": retained,
                  "executions": len(executions), "sample_count": len(valid),
                  "peak_bytes": max(row["accounted_peak_bytes"] for row in valid)}
    except BaseException as error:
        write_new(options.output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        stop.set()
        if observer is not None:
            observer.join(timeout=5)
        cleanup_error = None
        try:
            if started:
                rig.close(require_phone_execution=False)
        except BaseException as error:
            cleanup_error = error
        write_new(options.output / "CLEANUP.json", {
            "status": "PASS" if cleanup_error is None else "FAIL",
            "error": None if cleanup_error is None else repr(cleanup_error),
            "direct_phone_receipts": list(rig.direct_phone_receipts),
            "usb_restore_receipts": list(rig.usb_restore_receipts),
            "android_control_events": list(rig.android_control_events),
        })
        if cleanup_error is not None:
            raise cleanup_error
    write_new(options.output / "RESULT.json", result)


if __name__ == "__main__":
    main()
