"""Bounded native FFN calibration, not an online scheduling policy."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--library-dir", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--phone-shard", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", type=int, nargs="+", default=[0, 100, 75, 50, 25, 0])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--transport", choices=("tcp", "functionfs"), default="tcp")
    args = parser.parse_args()
    assert all(f in (0, 25, 50, 75, 100) for f in args.fractions)
    assert 1 <= args.repetitions <= 3
    args.output.mkdir()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.repo))
    from research_dev.scheduler.adapters.android_llama_server import (
        AndroidLlamaServerProcessConfiguration, AndroidLlamaServerProcessLauncher,
    )
    from research_dev.scheduler.adapters.host_runtime import (
        HostEnergySampler, default_host_metric_callbacks, server_energy_summary,
    )
    from research_dev.scheduler.adapters.http_backend import LlamaCppCompletionPayload, LlamaCppHttpClient
    from research_dev.scheduler.adapters.llama_server import LlamaServerLaunchContract, ManagedLlamaServer
    from research_dev.scheduler.adapters.probes import parse_android_process_identity, probe_android_phone_runtime
    from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeControl, AdaptiveDecodePolicy
    from research_dev.scheduler._internal.types import canonical_sha256
    from research_dev.scheduler.adapters.bridge import probe_functionfs_usb_device, verify_android_usb_restored
    from research_dev.scheduler.adapters.phone_session import DirectPhoneFfnSession, DirectPhoneFfnSessionConfiguration
    from research_dev.scheduler.adapters.phone_transport import PhoneTransportContract
    from research_dev.scheduler.adapters.llama_server import PhoneFfnExecutionContract
    from research_dev.scheduler.adapters.probes import probe_phone_runtime

    request = json.loads(args.request.read_text())
    index = json.loads(args.index.read_text())
    shard, = index["shards"]
    artifact = digest(args.model)
    assert artifact == index["parent_sha256"] == "sha256:" + request["model_artifact_sha256"]
    n_ff, n_embd = index["n_ff"], index["n_embd"]
    mask = int(shard["layer_mask"], 16)
    output_tokens = args.output_tokens or request["output_tokens"]
    save(args.output / "SPEC.json", {
        "scope": "native calibration; no scheduler qualification or whole-model NPU claim",
        "argv": sys.argv, "artifact_sha256": artifact, "index_sha256": digest(args.index),
        "shard": shard, "request": request, "tested_output_tokens": output_tokens,
        "server_sha256": digest(args.server), "probe_sha256": digest(Path(__file__)),
        "libraries": {p.name: digest(p) for p in args.server.parent.glob("*.so")},
        "cpu_parent": {"gpu_layers": 0, "threads": args.threads, "context": 4096,
                       "batch": 1024, "ubatch": 256, "parallel": 1},
        "phone_power": {"active_w": [3, 4.5, 6], "idle_w": 0.875},
    })
    remote_root = "/data/local/tmp/" + args.output.parent.name + "-" + args.output.name
    config = AndroidLlamaServerProcessConfiguration(
        adb_path=Path("/usr/bin/adb"), serial="3C15AU002CL00000", adb_port=5037,
        remote_server_path=args.worker, remote_library_directory=str(Path(args.worker).parent),
        remote_model_paths_by_artifact={artifact: args.phone_shard},
        remote_state_directory=remote_root, executable_device_name="HTP0", output_directory=args.output,
    )
    launcher = AndroidLlamaServerProcessLauncher(config)
    identity = None
    worker = None
    managed = None
    shared_endpoint = None
    shared_output = None
    shared_started = None
    shared_ready = None
    forwarded = False
    observer = None
    latest = []
    stop = threading.Event()
    commands = []
    sampler = HostEnergySampler(default_host_metric_callbacks())
    sampler.start()
    runs = []
    intervals = []
    pid_file = remote_root + "/worker.pid"
    worker_log = remote_root + "/worker.log"
    remote_port, local_port = 28761, 29761
    direct = None
    usb_transport = None
    diagnostic_endpoint = "http://192.168.42.1:18383"
    direct_started = False
    if args.transport == "functionfs":
        assert args.fractions[-1] > 0 and all(f > 0 for f in args.fractions[1:])
        usb_transport = PhoneTransportContract(
            transport="functionfs-usb", allocator="devmem", queue_depth=1, concurrent_streams=1,
            max_payload_bytes=n_embd * 2, full_duplex=True, split_h2d=False,
            generation="functionfs-dmabuf-ring-v1", profile_id="native-llama-calibration",
            usbfs_available_bytes=int(Path("/sys/module/usbcore/parameters/usbfs_memory_mb").read_text()) * 1024**2,
            slot_safety_bytes=65536, vendor_id=0x18d1, product_id=0x2d00,
            control_host="127.0.0.1", control_port=local_port,
        )
        direct = DirectPhoneFfnSession(DirectPhoneFfnSessionConfiguration(
            adb_path=config.adb_path, usb_close_path=Path("/home/zhihao/s42-mixed-residency-priority-deploy-20260828-v1/build-server-cuda/bin/llama-ffn-split-usb-close"),
            serial=config.serial, adb_port=config.adb_port,
            session_script="/data/local/tmp/s42-hal-runtime-probe-20260906-v4/direct_phone_ffn_session.sh",
            restore_script="/data/local/tmp/s42-unified-direct-v1/restore_android_usb.sh", session_root=remote_root,
            worker_paths_by_artifact={artifact: args.worker}, model_paths_by_artifact={artifact: args.phone_shard},
            backend_by_device={"op15": "HTP0"}, minimum_usb_speed_mbps=5000,
            required_kernel_release="6.12.23-android16-5-o-g227664cbe007-4k", launch_timeout_s=45,
            session_timeout_s=300, diagnostic_port=18383, diagnostic_host="192.168.42.1",
            busybox_path="/data/adb/magisk/busybox", network_manager_path=Path("/usr/bin/nmcli"),
            android_gadget_path="/config/usb_gadget/g1", functionfs_gadget_path="/config/usb_gadget/g2",
            functionfs_root_path="/dev/usb-ffs/s41", phone_usb_controller="a600000.dwc3",
        ))

    def shell(command, required=True):
        result = launcher._su(command, timeout_s=20, check=False)
        commands.append({"command": command, "time_ns": time.monotonic_ns(),
                         "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if required and result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout

    def check():
        if managed is not None and managed.process.poll() is not None:
            raise RuntimeError("Desktop server exited")
        if args.transport == "tcp" and worker is not None and worker.poll() is not None:
            raise RuntimeError("Phone worker exited: " + shell("tail -30 " + worker_log, False))
        if not latest or not latest[-1].to_json()["valid"] or latest[-1].value is None:
            raise RuntimeError("Phone health observation is unavailable")
        value = latest[-1].value
        if not value.thermal_qualified or value.temperature_millic >= 90000 or value.battery_ppm < 50000:
            raise RuntimeError("Phone thermal/battery check failed")
        if value.available_bytes < 805306368:
            raise RuntimeError("Phone memory reserve breached")

    def observe():
        with (args.output / "PHONE_HEALTH.jsonl").open("x") as stream:
            while not stop.is_set():
                observation = (probe_phone_runtime(diagnostic_endpoint, diagnostic=True) if direct_started else
                               probe_android_phone_runtime(config.serial, config.adb_port, diagnostic=True))
                latest[:] = [observation]
                row = observation.to_json()
                row["values"] = None if observation.value is None else asdict(observation.value)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                stop.wait(2)

    def run_fraction(sequence, fraction):
        nonlocal managed, shared_endpoint, shared_output, shared_started, shared_ready
        output = args.output / f"{sequence:02d}-fraction-{fraction}"
        output.mkdir()
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        environment = {k: v for k, v in os.environ.items()
                       if not k.startswith(("LLAMA_ARG_", "S41_SERVER_FFN_", "LLAMA_FFN_SPLIT_"))}
        environment.pop("GGML_CUDA_DISABLE_GRAPHS", None)
        environment["LD_LIBRARY_PATH"] = args.library_dir
        ffn = {}
        if fraction:
            ffn = {"S41_SERVER_FFN_HOST": "127.0.0.1", "S41_SERVER_FFN_PORT": str(local_port),
                   "S41_SERVER_FFN_ARTIFACT_SHA256": artifact,
                   "S41_SERVER_FFN_N_EMBD": str(n_embd), "S41_SERVER_FFN_LAYER_MASK": str(mask),
                   "S41_SERVER_FFN_COLUMNS": str(n_ff), "S41_SERVER_FFN_F16_IO": "1",
                   "S41_SERVER_FFN_ACTIVATION": "swiglu", "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
                   "S41_SERVER_FFN_MAX_TOKENS": "1"}
            environment.update(ffn)
            if usb_transport is not None:
                ffn.pop("S41_SERVER_FFN_HOST")
                ffn.pop("S41_SERVER_FFN_PORT")
                environment.pop("S41_SERVER_FFN_HOST")
                environment.pop("S41_SERVER_FFN_PORT")
                ffn.update(usb_transport.server_environment())
                environment.update(ffn)
        contract = LlamaServerLaunchContract(
            model_alias=request["model_id"], context_size=4096, batch_size=1024, ubatch_size=256,
            parallel=1, gpu_layers=0, cpu_device_id="desktop-cpu", gpu_device_id="desktop-gpu",
            phone_device_id="op15" if fraction else None, ffn_environment=ffn,
            threads=args.threads, threads_batch=args.threads,
        )
        command = (str(args.server), "--model", str(args.model), "--alias", request["model_id"],
                   "--device", "none", "--n-gpu-layers", "0", "--no-kv-offload", "--fit", "off",
                   "--ctx-size", "4096", "--batch-size", "1024", "--ubatch-size", "256",
                   "--parallel", "1", "--threads", str(args.threads), "--threads-batch", str(args.threads),
                   "--host", "127.0.0.1", "--port", str(port), "--no-webui", "--log-colors", "off")
        save(output / "LAUNCH.json", {"argv": command, "ffn_environment": ffn,
                                     "library_path": environment["LD_LIBRARY_PATH"]})
        reused = managed is not None
        if reused:
            port = shared_endpoint
            launched, ready = shared_started, shared_ready
            save(output / "REUSED_DESKTOP.json", {"port": port, "log_directory": str(shared_output),
                                                   "process_id": managed.process.pid})
        else:
            managed = ManagedLlamaServer(command, environment, output, "server", contract)
            launched = time.monotonic_ns()
            managed.start()
        try:
            deadline = time.monotonic() + 120
            while True:
                check()
                try:
                    if launcher._healthy(port):
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Desktop readiness exceeded 120 seconds")
                time.sleep(0.1)
            if not reused:
                ready = time.monotonic_ns()
                intervals.append((f"desktop_load_{sequence}", launched, ready))
                if fraction and direct is not None:
                    shared_endpoint, shared_output = port, output
                    shared_started, shared_ready = launched, ready
            for repetition in range(args.repetitions):
                first = []
                controls = []
                request_id = f"llama-split-{sequence}-{repetition}"

                def first_token(observed_ns):
                    first.append(observed_ns)
                    if not fraction:
                        return
                    desktop = {"artifact": artifact, "device": "desktop-cpu", "gpu_layers": 0,
                               "threads": args.threads, "context": 4096, "batch": 1024,
                               "ubatch": 256, "parallel": 1, "server_sha256": digest(args.server)}
                    plan = {"desktop": desktop, "layers": shard["layers"], "columns": n_ff * fraction // 100,
                            "shard_sha256": shard["shard_sha256"], "session_id": shard["session_id"],
                            "session_generation": 1, "activation": "swiglu", "io": "f16",
                            "transport": args.transport}
                    policy = AdaptiveDecodePolicy(
                        route_id="calibration-ffn-" + str(fraction), executor_id="native-calibration",
                        operator_plan_sha256=canonical_sha256(plan), desktop_parent_route_id="calibration-cpu",
                        desktop_placement_sha256=canonical_sha256(desktop), layer_indices=tuple(shard["layers"]),
                        layer_mask=mask, columns=plan["columns"], split_fraction_ppm=fraction * 10000,
                        resource_ids=("desktop-cpu", "op15-htp", "usb-" + args.transport),
                    )
                    control = AdaptiveDecodeControl(request_id, 0, repetition + 1, policy)
                    issued = time.monotonic_ns()
                    ack, received = LlamaCppHttpClient.apply_ffn_control(f"http://127.0.0.1:{port}", control)
                    assert ack["policy_hash"] == policy.policy_hash and ack["plan_generation"] == repetition + 1
                    assert ack["slot_id"] == 0
                    controls.append({"issued_ns": issued, "received_ns": received,
                                     "control": control.to_json(), "ack": ack})
                    save(output / f"CONTROL_{repetition}.json", controls[-1])

                payload = LlamaCppCompletionPayload(
                    request_id=request_id, expected_model_alias=request["model_id"],
                    input_tokens=request["input_tokens"], output_tokens=output_tokens,
                    prompt_tokens=tuple(request["prompt_tokens"]), seed=42, quality_mode="semantic", timeout_s=180,
                    stream_path=output / f"request-{repetition}.raw", on_first_token=first_token,
                )
                check()
                started = time.monotonic_ns()
                response = LlamaCppHttpClient().complete(f"http://127.0.0.1:{port}", payload, check)
                finished = time.monotonic_ns()
                row = {"sequence": sequence, "fraction": fraction, "repetition": repetition,
                       "started_ns": started, "finished_ns": finished, "first_token_ns": first[0],
                       "startup_s": 0 if reused else (ready - launched) / 1e9,
                       "wall_s": (finished - started) / 1e9,
                       "decode_tokens_per_s": output_tokens * 1000 / response["predicted_ms"],
                       "execution": response, "controls": controls}
                save(output / f"EXECUTION_{repetition}.json", row)
                runs.append(row)
                print("COMPLETED", fraction, repetition, row["wall_s"], row["decode_tokens_per_s"], flush=True)
        finally:
            if direct is None or fraction == 0 or sequence == len(args.fractions) - 1 or sys.exc_info()[0] is not None:
                managed.stop()
                managed = None
        if managed is not None:
            return
        log_output = shared_output if fraction and direct is not None else output
        terminal = [json.loads(line.split("S41SERVERFFN ", 1)[1]) for line in
                    (log_output / "server.stderr").read_text().splitlines() if "S41SERVERFFN {" in line]
        if fraction and (len(terminal) != 1 or terminal[0]["status"] != "ok" or terminal[0]["decode_calls"] <= 0):
            raise RuntimeError("Missing or failed nonzero native FFN proof")
        save(output / "TERMINAL.json", terminal)

    try:
        live = subprocess.check_output(["ps", "-eo", "comm="], text=True).splitlines()
        assert not any(name.strip() in ("llama-server", "llama-cli", "llama-bench") for name in live)
        phone_live = shell("ps -A -o PID,NAME; getprop sys.usb.config; uname -r")
        assert not any(line.split()[-1] in ("llama-server", "llama-ffn-split-worker", "llama-ffn-split-resident-workers")
                       for line in phone_live.splitlines() if line.split())
        measured = shell("sha256sum " + shlex.quote(args.phone_shard) + " " + shlex.quote(args.worker)
                         + " " + shlex.quote(config.remote_library_directory) + "/*.so")
        assert measured.split()[0] == shard["shard_sha256"].removeprefix("sha256:")
        save(args.output / "PHONE_HASHES.json", measured)
        latest[:] = [probe_android_phone_runtime(config.serial, config.adb_port, diagnostic=True)]
        check()
        assert shard["shard_bytes"] + 512 * 1024**2 + 805306368 <= latest[-1].value.available_bytes
        assert shard["shard_bytes"] + 512 * 1024**2 + 805306368 <= 10_000_000_000
        shell("mkdir " + remote_root)
        shell("awk '$2 ~ /:7059$/ && $4 == \"0A\" {busy=1} END {exit busy ? 1 : 0}' /proc/net/tcp /proc/net/tcp6")
        if direct is None:
            launcher._adb("forward", "--no-rebind", f"tcp:{local_port}", f"tcp:{remote_port}")
            forwarded = True
        arguments = [args.worker, "-m", args.phone_shard, "--artifact-sha256", artifact,
                     "--backend", "HTP0", "--layers", shard["layer_spec"], "--columns", str(n_ff),
                     "--column-quantum", str(n_ff // 4), "--max-tokens", "1", "--f16-io",
                     "--port", str(remote_port)]
        worker_environment = {"LD_LIBRARY_PATH": config.remote_library_directory,
                              "ADSP_LIBRARY_PATH": config.remote_library_directory,
                              "GGML_HEXAGON_NDEV": "1", "GGML_HEXAGON_NHVX": "4",
                              "GGML_HEXAGON_MBUF": "4192", "GGML_HEXAGON_VMEM": "3328",
                              "S41_DISABLE_GRAPH_CACHE": "1", "S42_RESIDENCY_SESSION_ID": "HTP0",
                              "S42_RESIDENCY_SESSION_GENERATION": "1"}
        body = ("echo $$ > " + pid_file + "; cd " + shlex.quote(config.remote_library_directory)
                + " && exec env " + " ".join(shlex.quote(k + "=" + v) for k, v in worker_environment.items())
                + " " + shlex.join(arguments) + " > " + worker_log + " 2>&1")
        if direct is not None:
            cfg = direct.configuration
            assert not shell("cat " + cfg.functionfs_gadget_path + "/UDC").strip()
            assert shell("uname -r").strip() == cfg.required_kernel_release
            worker_environment.update({
                "SCHEDULER_ANDROID_GADGET": cfg.android_gadget_path,
                "SCHEDULER_FUNCTIONFS_GADGET": cfg.functionfs_gadget_path,
                "SCHEDULER_FUNCTIONFS_ROOT": cfg.functionfs_root_path,
                "SCHEDULER_PHONE_UDC": cfg.phone_usb_controller,
                "S42_USB_NCM": "1", "S42_DIAGNOSTIC_PORT": str(cfg.diagnostic_port),
                "S42_BUSYBOX": cfg.busybox_path, "S41_FFN_F16_IO": "1",
                "S41_FFN_MAX_TOKENS": "1", "S41_FFN_COLUMN_QUANTUM": str(n_ff // 4),
                "S41_FFN_QUEUE_DEPTH": "1",
            })
            arguments = ["sh", cfg.session_script, args.worker, args.phone_shard,
                         shard["layer_spec"], str(n_ff), "HTP0", remote_root,
                         cfg.restore_script, "300", "0", artifact]
            body = ("nohup env " + " ".join(shlex.quote(k + "=" + v) for k, v in worker_environment.items())
                    + " " + shlex.join(arguments) + " > " + remote_root + "/session-launch.log 2>&1 < /dev/null &")
            save(args.output / "TRANSPORT_HASHES.json", shell("sha256sum " + cfg.session_script + " " + cfg.restore_script))
        save(args.output / "WORKER_LAUNCH.json", {"arguments": arguments, "environment": worker_environment})
        if direct is None:
            observer = threading.Thread(target=observe, daemon=True)
            observer.start()
        while len(sampler.rows()) < 2:
            time.sleep(0.1)
        started = time.monotonic_ns()
        worker = subprocess.Popen(launcher._adb_command(config.serial, "shell", "su -c " + shlex.quote(body)),
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 120
        while True:
            if direct is not None:
                try:
                    usb = probe_functionfs_usb_device(vendor_id="18d1", product_id="2d00")
                except Exception:
                    usb = None
                if usb is not None:
                    assert usb.negotiated_speed_mbps >= 5000
                    direct._connect_diagnostic_ncm(usb)
                    latest[:] = [probe_phone_runtime(diagnostic_endpoint, diagnostic=True)]
                    direct_started = True
                    check()
                    log = direct._read_diagnostic_file("residency.log")
                    assert "descriptors_ready backend=Hexagon" in log
                    save(args.output / "USB.json", vars(usb))
                    observer = threading.Thread(target=observe, daemon=True)
                    observer.start()
                    break
            pid = shell("cat " + pid_file, False).strip()
            if pid.isdecimal():
                identity = parse_android_process_identity(
                    shell(launcher._process_identity_command(int(pid), pid_file)), int(pid))
                assert identity.executable == args.worker
            if direct is None:
                check()
            log = shell("tail -8 " + worker_log, False)
            if direct is None and "[ffn-worker] ready " in log:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Phone worker not ready after 120 seconds")
            time.sleep(0.25)
        finished = time.monotonic_ns()
        intervals.append(("phone_preload", started, finished))
        save(args.output / "READY.json", {"started_ns": started, "ready_ns": finished,
                                         "process_identity": None if identity is None else identity.to_json(), "ready_log": log})
        print("PHONE_READY", (finished - started) / 1e9, flush=True)
        for sequence, fraction in enumerate(args.fractions):
            run_fraction(sequence, fraction)
        time.sleep(0.5)
        for row in runs:
            energy = server_energy_summary(sampler.rows_between(row["started_ns"], row["finished_ns"]),
                                           row["started_ns"], row["finished_ns"])
            energy["boundary"] = "request_service_including_prefill_and_decode"
            active_s = (row["finished_ns"] - row["first_token_ns"]) / 1e9 if row["fraction"] else 0
            energy["phone_active_s"] = active_s
            energy["fleet_j_by_phone_w"] = {str(power): energy["server_compute_device_energy_j"]
                + active_s * power + (row["wall_s"] - active_s) * 0.875 for power in (3, 4.5, 6)}
            row["energy"] = energy
        prepared = [{"kind": name, "start_ns": a, "end_ns": b,
                     "server_energy": server_energy_summary(sampler.rows_between(a, b), a, b)} for name, a, b in intervals]
        save(args.output / "RESULT.json", {"status": "PASS", "runs": runs, "preparation": prepared,
                                           "phone_weight_loads": 1, "scheduler_qualified": False})
    except BaseException as error:
        save(args.output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc(), "runs": runs})
        raise
    finally:
        stop.set()
        if observer is not None:
            observer.join(timeout=5)
        cleanup_errors = []
        if managed is not None:
            managed.stop()
        try:
            if direct is not None:
                try:
                    probe_functionfs_usb_device(vendor_id="18d1", product_id="2d00")
                except Exception:
                    pass
                else:
                    direct._close_direct_usb(PhoneFfnExecutionContract("op15", n_embd, tuple(shard["layers"]), mask,
                                                                       n_ff, 1, "swiglu"), usb_transport, artifact)
                verify_android_usb_restored(serial=config.serial, adb_port=config.adb_port,
                                            minimum_speed_mbps=5000, timeout_s=30)
            save(args.output / "PHONE_LOG.json", shell("cat " + worker_log, False))
            if identity is not None:
                launcher.stop_remote(identity.process_id, pid_file, expected=identity)
            if worker is not None:
                worker.wait(timeout=10)
            if forwarded:
                launcher.remove_forward(local_port)
        except Exception as error:
            cleanup_errors.append(repr(error))
        sampler.stop()
        save(args.output / "POWER_SAMPLES.json", sampler.rows())
        save(args.output / "POWER_DIAGNOSTICS.json", sampler.diagnostics())
        save(args.output / "COMMANDS.json", commands)
        save(args.output / "CLEANUP.json", {"errors": cleanup_errors,
                                          "status": "PASS" if not cleanup_errors else "FAIL"})
        if cleanup_errors:
            raise RuntimeError(cleanup_errors)


if __name__ == "__main__":
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("Calibration interrupted; stopping owned endpoints")
    signal.signal(signal.SIGTERM, interrupted)
    main()
