"""Bounded control-path diagnostic using one canonical offline preload stage."""

import argparse
import json
from pathlib import Path
import subprocess
import shlex
import shutil
import sys
import time
import threading
import traceback
from dataclasses import replace


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume-tcp-adb", action="store_true")
    parser.add_argument("--hash-cache-input", type=Path)
    parser.add_argument("--whole-inference", action="store_true")
    options = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(options.repo))
    from research_dev.scheduler.campaigns.burstgpt import runner
    from research_dev.scheduler.campaigns.burstgpt.offline_residency_gate import _SnapshotStore
    from research_dev.scheduler.adapters import CanonicalOfflinePhoneResidencyPreloader

    options.output.mkdir()
    for name in ("commands", "snapshots", "streams"):
        (options.output / name).mkdir()
    sequence = 0

    def command(arguments, *, timeout=10, required=True):
        nonlocal sequence
        started = time.monotonic_ns()
        try:
            result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout)
            value = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired as error:
            value = {"returncode": None, "error": repr(error)}
        value.update(command=arguments, started_ns=started, finished_ns=time.monotonic_ns())
        write_new(options.output / "commands" / f"{sequence:03d}.json", value)
        sequence += 1
        if required and value["returncode"] != 0:
            raise RuntimeError(value)
        return value

    original = json.loads((options.base_run / "RUN_COMMAND_EXECUTION.json").read_text())
    args = runner._build_parser().parse_args(original[2:])
    args.output = options.output
    args.phone_session_root = "/data/local/tmp/" + options.output.parent.name + "/" + options.output.name
    args.phone_whole_state_directory = args.phone_session_root + "-whole"
    args.phone_remote_hash_cache = options.output / "PHONE_HASH_CACHE.json"
    if options.hash_cache_input is not None:
        shutil.copyfile(options.hash_cache_input, args.phone_remote_hash_cache)
    models = runner._load_trace_models(args)
    scheduler, manifests, _ = runner._build_scheduler(args, models)
    if options.whole_inference:
        from research_dev.scheduler.adapters.android_llama_server import ncm_control_script_sha256
        from research_dev.scheduler.adapters.catalog_materialization import materialize_whole_model_endpoint
        args.phone_whole_control_transport = "adb-ncm"
        args.phone_whole_ncm_adb_endpoint = models.diagnostic_url.hostname + ":5555"
        manifest = manifests[models.llama_model_id]
        catalog = materialize_whole_model_endpoint(
            models.catalog, manifest, executor_id="physical:op15-phone", endpoint="http://127.0.0.1:29382",
            backend="android-llama-server-opencl", adapter_parameters={
                "batch_size": 1024, "context_size": 4096, "cpu_device_id": "desktop-cpu",
                "executable_device": args.phone_whole_executable_device,
                "execution_adapter": "android-llama-server-v1", "forward_port": 29382,
                "gpu_device_id": "op15-phone", "model_alias": manifest.model_id, "parallel": 1,
                "persistent_residency": 1, "whole_model_peak_memory_bytes": 3_000_000_000,
                "remote_library_directory": args.phone_whole_library_directory,
                "remote_model_path": args.phone_whole_model, "remote_port": 18382,
                "remote_server_path": args.phone_whole_server,
                "remote_server_sha256": "sha256:3fad5e2f2730d1e240f176010994c49dc4a6ea59fb970708df8d2a85bb0abe1c",
                "request_io_protocol": "token-ids-v1", "token_id_bytes": 4, "ubatch_size": 256,
                "android_control_transport": "adb-ncm",
                "android_control_endpoint": args.phone_whole_ncm_adb_endpoint,
                "android_control_script_sha256": ncm_control_script_sha256(),
            }, evidence_ids=(ncm_control_script_sha256(),), transition_latency_us=25_000_000,
            transition_energy_uj=125_000_000, transition_energy_maturity="SHADOW",
        )
        models = replace(models, catalog=catalog)
        scheduler, manifests, _ = runner._build_scheduler(args, models)
        write_new(options.output / "CATALOG.json", catalog.to_json())
    aliases, merged, _, _ = runner._select_replay(args, models, manifests)
    dependencies = dict(args.transport_host_dependency)
    rig = runner._build_rig(args, models, manifests, dependencies)
    snapshots = _SnapshotStore(options.output / "snapshots", rig)
    rig_started = False
    tcp_serial = models.diagnostic_url.hostname + ":5555"
    adb = [str(args.adb), "-P", str(args.adb_port)]
    bootstrap = adb + ["-s", args.phone_usb_serial]
    connected_here = False
    result_record = None
    before = command(bootstrap + ["shell", "cat /proc/sys/kernel/random/boot_id; getprop ro.serialno; uname -r"])
    processes = command(bootstrap + ["shell", "ps -A"])["stdout"]
    if any(value in processes for value in ("llama-", "ffn-split", "resident-router")):
        raise RuntimeError("Existing phone inference process; no mutation allowed")
    write_new(options.output / "DIRECT_PHONE_PREFLIGHT.json", rig.direct_phone_preflight().to_json())
    try:
        if options.resume_tcp_adb:
            control_root = args.phone_whole_state_directory
            command(bootstrap + ["shell", "su -c " + shlex.quote("mkdir " + shlex.quote(control_root))])
            source = options.repo / "research_dev/scheduler/adapters/native/android_ncm_adb_control.sh"
            remote = control_root + "/android_ncm_adb_control.sh"
            staging = "/data/local/tmp/s42-control-" + options.output.name + ".sh"
            command(bootstrap + ["push", str(source), staging])
            command(bootstrap + ["shell", "su -c " + shlex.quote("mv " + shlex.quote(staging) + " " + shlex.quote(remote))])
            words = ["sh", remote, control_root, args.phone_functionfs_gadget,
                     "5555", args.phone_usb_serial, "120"]
            body = "nohup " + shlex.join(words) + " > " + shlex.quote(control_root + "/control.log") + " 2>&1 < /dev/null &"
            command(bootstrap + ["shell", "su -c " + shlex.quote(body)])
            for _ in range(20):
                ready = command(bootstrap + ["shell", "su -c " + shlex.quote("cat " + control_root + "/control.ready")], required=False)
                if ready["returncode"] == 0 and ready["stdout"].strip() == "armed":
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("NCM TCP control did not arm")
        warm = runner._warm_payload(models, aliases, options.output / "streams")
        rig_started = True
        rig.start(warm)
        epoch_ns = time.monotonic_ns()
        rig.begin_offline_preload(epoch_ns)
        probe_item = next(row for row in merged if row["model_id"] == models.expected_gemma.model_id)
        from research_dev.scheduler.campaigns.burstgpt.offline_residency_gate import _request_probe
        probe = _request_probe(probe_item, aliases, options.output / "streams",
                               request_id="control-transport-preparation", arrival_us=1, seed=42)
        model_id = models.expected_gemma.model_id
        snapshot = snapshots.capture(probe.request, model_id, 1, "initial")
        plan = snapshots.plan(scheduler, {model_id: (probe.request,)}, snapshot, epoch_ns)
        write_new(options.output / "OFFLINE_PLAN.json", plan.to_json())
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler, rig.backend(), epoch_ns=epoch_ns, snapshot_provider=snapshots.offline,
        )
        stage = plan.current_stage
        now = (time.monotonic_ns() - epoch_ns) // 1000
        result = preloader.execute_next_stage(
            plan.plan_id, probe.payload, snapshot=snapshots.offline(stage, now), observed_at_us=now,
        )
        write_new(options.output / "PREPARATION.json", result.to_json())
        print("ONE_SESSION_READY", json.dumps(dict(rig.direct_phone_residency_state)), flush=True)
        if options.whole_inference:
            from research_dev.scheduler.campaigns.burstgpt.offline_residency_gate import _submit_request, _execution_json
            rig._android_phone_launcher.connect_ncm_control()
            health_samples = []
            for _ in range(40):
                _, health = rig.phone_runtime_observation()
                health_samples.append(health)
                if health.get("valid"):
                    break
                rig.request_runtime_observation_refresh()
                time.sleep(0.25)
            else:
                write_new(options.output / "HEALTH_PREFLIGHT.json", health_samples)
                raise RuntimeError("Whole-phone preflight lacks fresh health")
            write_new(options.output / "HEALTH_PREFLIGHT.json", health_samples)
            item = next(row for row in merged if row["model_id"] == models.llama_model_id)
            item = {**item, "row": {**item["row"], "output_tokens": 32}}
            whole = _request_probe(item, {**aliases, models.llama_model_id: manifest.model_id},
                                   options.output / "streams", request_id="whole-ncm-probe",
                                   arrival_us=(time.monotonic_ns() - epoch_ns) // 1000, seed=42)
            coordinator, ticket = _submit_request(scheduler, rig, snapshots, epoch_ns, whole, "calibration")
            print("WHOLE_SUBMITTED", ticket.decision.route_id, flush=True)
            before_whole = dict(rig.direct_phone_residency_state)
            during = []
            stop_observation = threading.Event()

            def observe_whole():
                while not stop_observation.is_set():
                    try:
                        snapshot = snapshots.capture(whole.request, models.llama_model_id,
                                                     (time.monotonic_ns() - epoch_ns) // 1000, "whole-active")
                        during.append(snapshot.to_json()["telemetry_observations"]["op15-phone"])
                    except Exception as error:
                        during.append({"error": repr(error)})
                    stop_observation.wait(0.5)

            observer = threading.Thread(target=observe_whole)
            observer.start()
            try:
                completed = coordinator.drain(timeout_s=180)
            finally:
                stop_observation.set()
                observer.join(timeout=5)
                coordinator.close()
                write_new(options.output / "WHOLE_ACTIVE_TELEMETRY.json", during)
            execution = completed.executions[whole.request.request_id]
            write_new(options.output / "WHOLE_EXECUTION.json", _execution_json(execution))
            if execution.command.adapter_parameters.get("execution_adapter") != "android-llama-server-v1":
                raise RuntimeError("Calibration did not select the whole-phone route; no route forced")
            if execution.recoveries:
                raise RuntimeError("Whole-phone execution used fallback")
            samples = []
            for _ in range(12):
                snapshot = snapshots.capture(whole.request, models.llama_model_id,
                                             (time.monotonic_ns() - epoch_ns) // 1000, "whole-memory")
                samples.append(snapshot.to_json()["telemetry_observations"]["op15-phone"])
                time.sleep(0.5)
            write_new(options.output / "WHOLE_TELEMETRY.json", samples)
            if not all(any(row.get("validity") == "VALID" for row in sample.get("resident_allocations", []))
                       for sample in samples[-6:]):
                raise RuntimeError("Whole-phone allocation samples are not fresh and generation-bound")
            after_whole = dict(rig.direct_phone_residency_state)
            if any(before_whole[key] != after_whole[key] for key in ("phone_shards", "load_count_by_session")):
                raise RuntimeError("Whole-phone execution changed retained HTP residency")
            write_new(options.output / "CONTROL_EVENTS.json", rig._android_phone_launcher.control_events)
            print("WHOLE_EXECUTION_COMPLETE", flush=True)
        command(adb + ["devices", "-l"])
        for _ in range(10):
            connect = command(adb + ["connect", tcp_serial], required=False)
            connected = connect["returncode"] == 0 and connect.get("stdout", "").strip() in {
                "connected to " + tcp_serial, "already connected to " + tcp_serial,
            }
            connected_here = connected and connect["stdout"].strip() == "connected to " + tcp_serial
            if connected:
                break
            time.sleep(0.5)
        print("NCM_ADB_CONNECT", json.dumps(connect), flush=True)
        if not connected:
            raise RuntimeError("Existing TCP ADB is unavailable during FunctionFS")
        network = adb + ["-s", tcp_serial]
        identity = command(network + ["shell", "cat /proc/sys/kernel/random/boot_id; getprop ro.serialno; uname -r"])
        if identity["stdout"] != before["stdout"]:
            raise RuntimeError("NCM ADB identifies a different boot or phone")
        samples = []
        for index in range(12):
            value = command(network + ["shell", "su -c 'cat /proc/uptime; cat /proc/meminfo; cat /config/usb_gadget/g2/UDC; ps -A -o PID,NAME'"])
            samples.append(value)
            time.sleep(0.5)
        result_record = {
            "status": "PASS", "scope": "whole-model inference and control" if options.whole_inference else "control only; no whole-model inference proof",
            "bootstrap_identity": before, "ncm_identity": identity,
            "sample_count": len(samples), "session_state": dict(rig.direct_phone_residency_state),
            "phone_phase_events": list(rig.phone_residency_phase_events),
        }
    except BaseException as error:
        write_new(options.output / "FAILURE.json", {
            "error": repr(error), "traceback": traceback.format_exc(),
            "phone_state": dict(rig.direct_phone_residency_state),
            "android_control_events": list(rig.android_control_events),
        })
        raise
    finally:
        cleanup_error = None
        try:
            if rig_started:
                rig.close(require_phone_execution=False)
        except BaseException as error:
            cleanup_error = error
        if connected_here:
            command(adb + ["disconnect", tcp_serial], required=False)
        write_new(options.output / "CLEANUP.json", {
            "status": "PASS" if cleanup_error is None else "FAIL",
            "error": None if cleanup_error is None else repr(cleanup_error),
            "direct_phone_receipts": list(rig.direct_phone_receipts),
            "usb_restore_receipts": list(rig.usb_restore_receipts),
            "android_control_events": list(rig.android_control_events),
        })
        if cleanup_error is not None:
            if not (options.output / "FAILURE.json").exists():
                write_new(options.output / "FAILURE.json", {"error": repr(cleanup_error), "phase": "cleanup"})
            raise cleanup_error
        if result_record is not None:
            write_new(options.output / "RESULT.json", result_record)


if __name__ == "__main__":
    main()
