"""Gemma decode split-ratio sweep: phone share of FFN columns versus the desktop, decode only.

Native calibration on the real OP15 over FunctionFS with one HTP session (the HTP0 shard,
layers 0-7 at full width) and the calibrated CUDA desktop parent (23 GPU layers). Runtime
control is applied at the first generated token, so every prefill runs locally and only
decode microbatches split the FFN columns of the session's layers between the desktop
prefix and the phone suffix. Fractions are the phone's percentage of FFN columns.

This is a bounded measurement, not a scheduler qualification: no leases, no transport
identity, one session of three, direct worker path (no resident router).
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

SERIAL = "3C15AU002CL00000"
VENDOR, PRODUCT = 0x18d1, 0x2d00


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=1)
        stream.write("\n")


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def layer_spec(mask):
    indices = [i for i in range(64) if mask >> i & 1]
    spans, first, previous = [], indices[0], indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        spans.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = index
    spans.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(spans), indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True, help="DOCUMENT_REQUESTS.json with prompt tokens")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--shard-remote-dir", required=True)
    parser.add_argument("--session-id", default="HTP0")
    parser.add_argument("--worker", required=True)
    parser.add_argument("--usb-close", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", type=int, nargs="+", default=[0, 100, 75, 50, 25, 0])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=96)
    parser.add_argument("--gpu-layers", type=int, default=23)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--column-quantum", type=int, default=1280)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--owner-timeout-ms", type=int, default=120000)
    parser.add_argument("--kernel-release", default="6.12.23-android16-5-o-g227664cbe007-4k")
    args = parser.parse_args()
    assert all(0 <= f <= 100 for f in args.fractions) and 1 <= args.repetitions <= 3
    args.output.mkdir()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.repo))
    from research_dev.scheduler import ModelManifest
    from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeControl, AdaptiveDecodePolicy
    from research_dev.scheduler._internal.types import canonical_sha256
    from research_dev.scheduler.adapters import (
        HostEnergySampler, LlamaCppHttpClient, LlamaServerLaunchContract, LlamaServerProcessConfiguration,
        LlamaServerProcessLauncher, default_host_metric_callbacks, server_energy_summary,
    )
    from research_dev.scheduler.adapters.android_llama_server import (
        AndroidLlamaServerProcessConfiguration, AndroidLlamaServerProcessLauncher,
    )
    from research_dev.scheduler.adapters.bridge import probe_functionfs_usb_device, verify_android_usb_restored
    from research_dev.scheduler.adapters.ffn_shards import FfnShardIndex
    from research_dev.scheduler.adapters.http_backend import LlamaCppCompletionPayload
    from research_dev.scheduler.adapters.llama_server import PhoneFfnExecutionContract
    from research_dev.scheduler.adapters.phone_session import DirectPhoneFfnSession, DirectPhoneFfnSessionConfiguration
    from research_dev.scheduler.adapters.phone_transport import PhoneTransportContract
    from research_dev.scheduler.adapters.probes import parse_android_process_identity, probe_phone_runtime

    manifest = ModelManifest.from_json(json.loads(args.manifest.read_text()))
    artifact = manifest.artifact_sha256
    n_ff, n_embd = manifest.feed_forward_length, manifest.embedding_length
    quantum = args.column_quantum
    assert n_ff % quantum == 0 and all(n_ff * f // 100 % quantum == 0 for f in args.fractions)
    index = FfnShardIndex.load(args.index, args.shard_remote_dir)
    assert index.parent_sha256 == artifact
    shard = None
    for candidate_mask in (0xff, 0xff00, 0xff0000):
        shard = index.resolve(artifact, candidate_mask, n_ff, session_id=args.session_id)
        if shard is not None:
            break
    assert shard is not None and shard.columns == n_ff, "session shard must be complete width"
    mask = shard.layer_mask
    layers, layer_indices = layer_spec(mask)
    requests = json.loads(args.requests.read_text())
    row = requests["rows"][0]
    prompt_tokens = tuple(int(v) for v in row["prompt_tokens"])
    alias = manifest.model_id
    activation = "geglu" if manifest.architecture == "gemma4" else "swiglu"
    payload_bytes = n_embd * 2 * args.max_tokens
    save(args.output / "SPEC.json", {
        "scope": "decode split-ratio calibration; one HTP session; not a scheduler qualification",
        "argv": sys.argv, "artifact_sha256": artifact, "index_sha256": index.index_sha256,
        "shard": {"session_id": shard.session_hint, "remote_path": shard.remote_path, "layer_mask": mask,
                  "layers": layers, "columns": shard.columns, "shard_bytes": shard.shard_bytes,
                  "shard_sha256": shard.shard_sha256},
        "prompt": {"input_tokens": len(prompt_tokens), "prompt_sha256": requests.get("prompt_sha256"),
                   "output_tokens": args.output_tokens},
        "desktop": {"gpu_layers": args.gpu_layers, "context": args.context, "batch": args.batch,
                    "ubatch": args.ubatch, "parallel": 1, "server_sha256": digest(args.server),
                    "libraries": {p.name: digest(p) for p in sorted(args.server.parent.glob("*.so*")) if p.is_file() and not p.is_symlink()}},
        "phone": {"worker": args.worker, "max_tokens": args.max_tokens, "column_quantum": quantum,
                  "transport": "functionfs-dmabuf-async-ring-v2", "queue_depth": 4, "payload_bytes": payload_bytes},
        "fractions_percent": args.fractions, "repetitions": args.repetitions,
        "phone_power_assumed_w": {"active": [3, 4.5, 6], "idle": 0.875},
    })
    remote_root = "/data/local/tmp/" + args.output.parent.name + "-" + args.output.name
    config = AndroidLlamaServerProcessConfiguration(
        adb_path=Path("/usr/bin/adb"), serial=SERIAL, adb_port=5037,
        remote_server_path=args.worker, remote_library_directory=str(Path(args.worker).parent),
        remote_model_paths_by_artifact={artifact: shard.remote_path},
        remote_state_directory=remote_root, executable_device_name="HTP0", output_directory=args.output,
    )
    android = AndroidLlamaServerProcessLauncher(config)
    usb_transport = PhoneTransportContract(
        transport="functionfs-usb", allocator="devmem", queue_depth=4, concurrent_streams=4,
        max_payload_bytes=payload_bytes, full_duplex=True, split_h2d=False,
        generation="functionfs-dmabuf-async-ring-v2", profile_id="native-gemma-decode-split-calibration",
        usbfs_available_bytes=int(Path("/sys/module/usbcore/parameters/usbfs_memory_mb").read_text()) * 1024**2,
        slot_safety_bytes=65536, vendor_id=VENDOR, product_id=PRODUCT,
        control_host="127.0.0.1", control_port=0, batch_plan="split-row",
    )
    direct = DirectPhoneFfnSession(DirectPhoneFfnSessionConfiguration(
        adb_path=config.adb_path, usb_close_path=args.usb_close, serial=SERIAL, adb_port=5037,
        session_script="/data/local/tmp/s42-hal-runtime-probe-20260906-v4/direct_phone_ffn_session.sh",
        restore_script="/data/local/tmp/s42-unified-direct-v1/restore_android_usb.sh", session_root=remote_root,
        worker_paths_by_artifact={artifact: args.worker}, model_paths_by_artifact={artifact: shard.remote_path},
        backend_by_device={"op15-phone": "HTP0"}, minimum_usb_speed_mbps=5000,
        required_kernel_release=args.kernel_release, launch_timeout_s=60, session_timeout_s=7200,
        diagnostic_port=18383, diagnostic_host="192.168.42.1", busybox_path="/data/adb/magisk/busybox",
        network_manager_path=Path("/usr/bin/nmcli"), android_gadget_path="/config/usb_gadget/g1",
        functionfs_gadget_path="/config/usb_gadget/g2", functionfs_root_path="/dev/usb-ffs/s41",
        phone_usb_controller="a600000.dwc3",
    ))
    cfg = direct.configuration
    sampler = HostEnergySampler(default_host_metric_callbacks())
    client = LlamaCppHttpClient()
    stop = threading.Event()
    commands, runs, intervals, latest = [], [], [], []
    managed = None
    worker_process = None
    identity = None
    pid_file, worker_log = remote_root + "/worker.pid", remote_root + "/worker.log"
    diagnostic_endpoint = "http://192.168.42.1:18383"
    direct_started = False
    generation = 0

    def shell(command, required=True):
        result = android._su(command, timeout_s=30, check=False)
        commands.append({"command": command, "time_ns": time.monotonic_ns(), "returncode": result.returncode,
                         "stdout": result.stdout[-4000:], "stderr": result.stderr[-2000:]})
        if required and result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout

    def check():
        if managed is not None and managed.process is not None and managed.process.poll() is not None:
            raise RuntimeError("desktop server exited")
        if latest and latest[-1].value is not None:
            value = latest[-1].value
            if not value.thermal_qualified or value.temperature_millic >= 90000 or value.battery_ppm < 50000:
                raise RuntimeError("phone thermal/battery check failed")

    def observe_phone():
        with (args.output / "PHONE_HEALTH.jsonl").open("x") as stream:
            while not stop.is_set():
                try:
                    observation = probe_phone_runtime(diagnostic_endpoint, diagnostic=True)
                    latest[:] = [observation]
                    entry = observation.to_json()
                    entry["values"] = None if observation.value is None else asdict(observation.value)
                    stream.write(json.dumps(entry) + "\n")
                    stream.flush()
                except Exception as error:  # noqa: BLE001 - recorded
                    stream.write(json.dumps({"error": repr(error), "time_ns": time.monotonic_ns()}) + "\n")
                stop.wait(2)

    def observe_server_memory():
        with (args.output / "SERVER_MEMORY.jsonl").open("x") as stream:
            while not stop.is_set():
                server = managed
                if server is not None and server.process is not None and server.process.poll() is None:
                    pid = server.process.pid
                    try:
                        status = Path(f"/proc/{pid}/status").read_text()
                        fields = {line.split(":", 1)[0]: int(line.split()[1]) * 1024 for line in status.splitlines()
                                  if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "RssFile:", "RssAnon:"))}
                        stream.write(json.dumps({"observed_ns": time.monotonic_ns(), "pid": pid, "bytes": fields}) + "\n")
                        stream.flush()
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                stop.wait(0.25)

    launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(
        server_path=args.server, model_paths_by_artifact={artifact: args.model},
        library_paths_by_device={"desktop-cuda": (args.cuda_lib_dir,)}, executable_device_names={"desktop-cuda": "CUDA0"},
        output_directory=args.output, common_library_paths=(args.server.parent, args.cuda_lib_dir),
    ))

    def ffn_environment():
        environment = {
            "S41_SERVER_FFN_ACTIVATION": activation, "S41_SERVER_FFN_ARTIFACT_SHA256": artifact,
            "S41_SERVER_FFN_COLUMNS": str(n_ff), "S41_SERVER_FFN_F16_IO": "1",
            "S41_SERVER_FFN_LAYER_MASK": str(mask), "S41_SERVER_FFN_MAX_TOKENS": str(args.max_tokens),
            "S41_SERVER_FFN_N_EMBD": str(n_embd), "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
            "S41_SERVER_FFN_TIMEOUT_MS": str(args.owner_timeout_ms),
        }
        environment.update(usb_transport.server_environment())
        return environment

    def launch_desktop(label, with_phone):
        nonlocal managed
        with socket.socket() as probe_socket:
            probe_socket.bind(("127.0.0.1", 0))
            port = probe_socket.getsockname()[1]
        contract = LlamaServerLaunchContract(
            model_alias=alias, context_size=args.context, parallel=1, batch_size=args.batch, ubatch_size=args.ubatch,
            gpu_layers=args.gpu_layers, cpu_device_id="desktop-cpu", gpu_device_id="desktop-cuda",
            phone_device_id="op15-phone" if with_phone else None,
            ffn_environment=ffn_environment() if with_phone else {},
            cuda_graph_mode="default", desktop_launch_mode="runtime-defaults",
        )
        endpoint = f"http://127.0.0.1:{port}"
        started = time.monotonic_ns()
        managed = launcher.launch_contract(endpoint, contract, manifest, label=label, control_check=check)
        ready = time.monotonic_ns()
        intervals.append({"kind": "desktop_load_" + label, "start_ns": started, "end_ns": ready,
                          "with_phone": with_phone, "ffn_environment": dict(contract.ffn_environment)})
        return endpoint

    def run_request(endpoint, fraction, sequence, repetition):
        request_id = f"gemma-split-{sequence}-{fraction}-{repetition}"
        first, controls = [], []

        def first_token(observed_ns):
            nonlocal generation
            first.append(observed_ns)
            if not fraction:
                return
            generation += 1
            columns = n_ff * fraction // 100
            desktop = {"artifact": artifact, "gpu_layers": args.gpu_layers, "context": args.context,
                       "batch": args.batch, "ubatch": args.ubatch, "server_sha256": digest(args.server)}
            plan = {"desktop": desktop, "layers": layer_indices, "columns": columns, "shard_sha256": shard.shard_sha256,
                    "session_id": shard.session_hint, "activation": activation, "io": "f16",
                    "transport": "functionfs", "column_quantum": quantum}
            policy = AdaptiveDecodePolicy(
                route_id=f"calibration-decode-split-{fraction}", executor_id="native-calibration",
                operator_plan_sha256=canonical_sha256(plan), desktop_parent_route_id="calibration-cuda-23",
                desktop_placement_sha256=canonical_sha256(desktop), layer_indices=tuple(layer_indices),
                layer_mask=mask, columns=columns, split_fraction_ppm=fraction * 10000,
                resource_ids=("desktop-cpu", "desktop-cuda", "op15-htp", "op15-functionfs"),
            )
            control = AdaptiveDecodeControl(request_id, 0, generation, policy)
            issued = time.monotonic_ns()
            ack, received = LlamaCppHttpClient.apply_ffn_control(endpoint, control)
            controls.append({"issued_ns": issued, "received_ns": received, "control": control.to_json(), "ack": ack})

        payload = LlamaCppCompletionPayload(
            request_id=request_id, expected_model_alias=alias, input_tokens=len(prompt_tokens),
            output_tokens=args.output_tokens, prompt_tokens=prompt_tokens, seed=42, quality_mode="semantic",
            timeout_s=1800, stream_path=args.output / f"{request_id}.raw", on_first_token=first_token,
        )
        check()
        started = time.monotonic_ns()
        response = client.complete(endpoint, payload, check)
        finished = time.monotonic_ns()
        timings = response.get("timings") or {}
        predicted_ms = float(response.get("predicted_ms") or timings.get("predicted_ms") or 0.0)
        predicted_n = int(response.get("predicted_n") or timings.get("predicted_n") or 0)
        prompt_ms = float(response.get("prompt_ms") or timings.get("prompt_ms") or 0.0)
        entry = {"sequence": sequence, "fraction_percent": fraction, "repetition": repetition, "request_id": request_id,
                 "started_ns": started, "finished_ns": finished, "first_token_ns": first[0] if first else None,
                 "wall_s": (finished - started) / 1e9, "prompt_ms": prompt_ms, "predicted_ms": predicted_ms,
                 "predicted_n": predicted_n,
                 "decode_ms_per_token": predicted_ms / predicted_n if predicted_n else None,
                 "tokens": list(response.get("tokens", [])), "controls": controls,
                 "result_keys": sorted(response.keys())}
        save(args.output / f"EXECUTION-{request_id}.json", {**entry, "execution": {k: v for k, v in response.items() if k != "tokens"}})
        runs.append(entry)
        print("COMPLETED", fraction, repetition, f"wall={entry['wall_s']:.2f}s decode={entry['decode_ms_per_token']}", flush=True)

    try:
        sampler.start()
        memory_observer = threading.Thread(target=observe_server_memory, daemon=True)
        memory_observer.start()
        live = subprocess.check_output(["ps", "-eo", "comm="], text=True).splitlines()
        assert not any(name.strip() in ("llama-server", "llama-cli") for name in live), "desktop server already running"
        phone_live = shell("ps -A -o PID,NAME; getprop sys.usb.config; uname -r")
        assert not any(line.split()[-1] in ("llama-server", "llama-ffn-split-worker", "llama-ffn-split-resident-workers",
                                            "llama-ffn-split-resident-router") for line in phone_live.splitlines() if line.split()), "phone worker already running"
        assert shell("uname -r").strip() == cfg.required_kernel_release, "phone kernel is not the qualified candidate"
        assert not shell("cat " + cfg.functionfs_gadget_path + "/UDC").strip(), "FunctionFS gadget already bound"
        measured = shell("sha256sum " + shlex.quote(shard.remote_path) + " " + shlex.quote(args.worker))
        assert measured.split()[0] == shard.shard_sha256.removeprefix("sha256:"), "phone shard digest differs"
        save(args.output / "PHONE_HASHES.json", measured)
        shell("mkdir " + shlex.quote(remote_root))
        worker_environment = {
            "LD_LIBRARY_PATH": config.remote_library_directory, "ADSP_LIBRARY_PATH": config.remote_library_directory,
            "GGML_HEXAGON_NDEV": "1", "GGML_HEXAGON_NHVX": "4", "GGML_HEXAGON_MBUF": "4192", "GGML_HEXAGON_VMEM": "3328",
            "S41_DISABLE_GRAPH_CACHE": "1", "S42_RESIDENCY_SESSION_ID": shard.session_hint,
            "S42_RESIDENCY_SESSION_GENERATION": "1",
            "SCHEDULER_ANDROID_GADGET": cfg.android_gadget_path, "SCHEDULER_FUNCTIONFS_GADGET": cfg.functionfs_gadget_path,
            "SCHEDULER_FUNCTIONFS_ROOT": cfg.functionfs_root_path, "SCHEDULER_PHONE_UDC": cfg.phone_usb_controller,
            "S42_USB_NCM": "1", "S42_DIAGNOSTIC_PORT": str(cfg.diagnostic_port), "S42_BUSYBOX": cfg.busybox_path,
            "S41_FFN_F16_IO": "1", "S41_FFN_STAGED_DMABUF": "0", "S41_FFN_MAX_TOKENS": str(args.max_tokens),
            "S41_FFN_COLUMN_QUANTUM": str(quantum), "S41_FFN_QUEUE_DEPTH": str(usb_transport.queue_depth),
        }
        arguments = ["sh", cfg.session_script, args.worker, shard.remote_path, layers, str(n_ff), "HTP0",
                     remote_root, cfg.restore_script, str(cfg.session_timeout_s), "0", artifact]
        body = ("nohup env " + " ".join(shlex.quote(k + "=" + v) for k, v in worker_environment.items())
                + " " + shlex.join(arguments) + " > " + remote_root + "/session-launch.log 2>&1 < /dev/null &")
        save(args.output / "WORKER_LAUNCH.json", {"arguments": arguments, "environment": worker_environment})
        while len(sampler.rows()) < 2:
            time.sleep(0.1)
        started = time.monotonic_ns()
        worker_process = subprocess.Popen(android._adb_command(SERIAL, "shell", "su -c " + shlex.quote(body)),
                                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 180
        while True:
            try:
                usb = probe_functionfs_usb_device(vendor_id=f"{VENDOR:04x}", product_id=f"{PRODUCT:04x}")
            except Exception:  # noqa: BLE001 - not enumerated yet
                usb = None
            if usb is not None:
                assert usb.negotiated_speed_mbps >= 5000, usb
                direct._connect_diagnostic_ncm(usb)
                latest[:] = [probe_phone_runtime(diagnostic_endpoint, diagnostic=True)]
                direct_started = True
                residency = direct._read_diagnostic_file("residency.log")
                assert "descriptors_ready backend=Hexagon" in residency, residency[-2000:]
                save(args.output / "USB.json", vars(usb))
                break
            terminal = shell("cat " + remote_root + "/terminal.status 2>/dev/null", False).strip()
            if terminal:
                raise RuntimeError("phone session terminated before USB enumeration: " + terminal + "\n"
                                   + shell("tail -40 " + remote_root + "/session-launch.log " + worker_log, False))
            if time.monotonic() > deadline:
                raise TimeoutError("phone session not enumerated after 180 s\n" + shell("tail -40 " + worker_log, False))
            time.sleep(0.5)
        pid = shell("cat " + pid_file, False).strip()
        if pid.isdecimal():
            identity = parse_android_process_identity(shell(android._process_identity_command(int(pid), pid_file)), int(pid))
        ready = time.monotonic_ns()
        intervals.append({"kind": "phone_preload", "start_ns": started, "end_ns": ready})
        save(args.output / "READY.json", {"started_ns": started, "ready_ns": ready,
                                         "process_identity": None if identity is None else identity.to_json()})
        print("PHONE_READY", (ready - started) / 1e9, flush=True)
        phone_observer = threading.Thread(target=observe_phone, daemon=True)
        phone_observer.start()
        check()

        shared_endpoint = None
        for sequence, fraction in enumerate(args.fractions):
            if fraction == 0:
                if managed is not None:
                    managed.stop()
                    managed = None
                    shared_endpoint = None
                    time.sleep(1.0)
                endpoint = launch_desktop(f"plain-{sequence}", with_phone=False)
                for repetition in range(args.repetitions):
                    run_request(endpoint, 0, sequence, repetition)
                managed.stop()
                managed = None
                time.sleep(1.0)
                continue
            if shared_endpoint is None:
                shared_endpoint = launch_desktop(f"assisted-{sequence}", with_phone=True)
            for repetition in range(args.repetitions):
                run_request(shared_endpoint, fraction, sequence, repetition)
        if managed is not None:
            managed.stop()
            managed = None
        time.sleep(0.5)
        for entry in runs:
            energy = server_energy_summary(sampler.rows_between(entry["started_ns"], entry["finished_ns"]),
                                           entry["started_ns"], entry["finished_ns"])
            energy = dict(energy)
            decode_s = ((entry["finished_ns"] - entry["first_token_ns"]) / 1e9) if entry["first_token_ns"] else None
            energy["phone_active_s"] = decode_s if entry["fraction_percent"] else 0.0
            energy["boundary"] = "request_service_including_local_prefill_and_decode"
            entry["energy"] = energy
        worker_tail = shell("tail -20 " + worker_log, False)
        summary = {}
        for fraction in sorted({r["fraction_percent"] for r in runs}):
            rows = [r for r in runs if r["fraction_percent"] == fraction and r["decode_ms_per_token"]]
            summary[str(fraction)] = {
                "runs": len(rows),
                "decode_ms_per_token_mean": sum(r["decode_ms_per_token"] for r in rows) / len(rows) if rows else None,
                "decode_ms_per_token_min": min(r["decode_ms_per_token"] for r in rows) if rows else None,
                "prompt_ms_mean": sum(r["prompt_ms"] for r in rows) / len(rows) if rows else None,
                "wall_s_mean": sum(r["wall_s"] for r in rows) / len(rows) if rows else None,
                "server_energy_j_mean": sum(r["energy"]["server_compute_device_energy_j"] for r in rows) / len(rows) if rows else None,
            }
        save(args.output / "RESULT.json", {"status": "PASS", "runs": runs, "intervals": intervals, "summary": summary,
                                           "worker_log_tail": worker_tail, "phone_weight_loads": 1,
                                           "scheduler_qualified": False})
        print(json.dumps(summary, indent=1), flush=True)
    except BaseException as error:
        save(args.output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc(), "runs": runs})
        raise
    finally:
        stop.set()
        cleanup_errors = []
        if managed is not None:
            try:
                managed.stop()
            except Exception as error:  # noqa: BLE001
                cleanup_errors.append("server stop: " + repr(error))
        try:
            if direct_started:
                for _attempt in range(30):
                    try:
                        probe_functionfs_usb_device(vendor_id=f"{VENDOR:04x}", product_id=f"{PRODUCT:04x}")
                    except Exception:  # noqa: BLE001
                        state = android._adb("get-state", check=False, timeout_s=2).stdout.strip()
                        if state == "device":
                            break
                        time.sleep(0.2)
                    else:
                        direct._close_direct_usb(PhoneFfnExecutionContract("op15-phone", n_embd, tuple(layer_indices), mask,
                                                                           n_ff, args.max_tokens, activation), usb_transport, artifact)
                        break
                verify_android_usb_restored(serial=SERIAL, adb_port=5037, minimum_speed_mbps=5000, timeout_s=60)
            save(args.output / "PHONE_LOG.json", shell("cat " + worker_log, False))
            save(args.output / "SESSION_LAUNCH_LOG.json", shell("cat " + remote_root + "/session-launch.log", False))
            if identity is not None:
                android.stop_remote(identity.process_id, pid_file, expected=identity)
            if worker_process is not None:
                worker_process.wait(timeout=15)
        except Exception as error:  # noqa: BLE001
            cleanup_errors.append(repr(error))
        sampler.stop()
        save(args.output / "POWER_SAMPLES.json", sampler.rows())
        save(args.output / "COMMANDS.json", commands)
        save(args.output / "CLEANUP.json", {"errors": cleanup_errors, "status": "PASS" if not cleanup_errors else "FAIL"})
        if cleanup_errors:
            print("CLEANUP_ERRORS", cleanup_errors, flush=True)


if __name__ == "__main__":
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("sweep interrupted; stopping owned endpoints")
    signal.signal(signal.SIGTERM, interrupted)
    main()
