"""Standalone Llama speed diagnostic; no scheduling or qualification policy."""

import argparse
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import sys
import threading
import time
import traceback


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--base-execution", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-layers", type=int, default=16)
    parser.add_argument("--flash-attn", choices=("off", "on"), default="off")
    parser.add_argument("--repetitions", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--serial", default="3C15AU002CL00000")
    options = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(options.repo))
    from research_dev.scheduler.adapters.android_llama_server import (
        AndroidLlamaServerProcessConfiguration, AndroidLlamaServerProcessLauncher,
    )
    from research_dev.scheduler.adapters.llama_server import ManagedLlamaServer, LlamaServerLaunchContract
    from research_dev.scheduler.adapters.http_backend import LlamaCppCompletionPayload, LlamaCppHttpClient
    from research_dev.scheduler.adapters.probes import parse_android_process_identity, probe_android_phone_runtime

    options.output.mkdir()
    shutil.copyfile(Path(__file__), options.output / "probe.py")
    (options.output / "commands").mkdir()
    base = json.loads(options.base_execution.read_text())["command"]
    parameters = base["adapter_parameters"]
    requests = [json.loads(line) for line in options.requests.read_text().splitlines() if line.strip()]
    request, = [row for row in requests if row["combined_request_index"] == 37]
    assert request["input_tokens"] == len(request["prompt_tokens"]) == 915
    assert request["output_tokens"] == 292
    assert "sha256:" + request["model_artifact_sha256"] == base["artifact_sha256"]
    write_new(options.output / "REQUEST.json", request)
    write_new(options.output / "INVOCATION.json", sys.argv)
    config = AndroidLlamaServerProcessConfiguration(
        adb_path=Path("/usr/bin/adb"), serial=options.serial, adb_port=5037,
        remote_server_path=parameters["remote_server_path"],
        remote_library_directory=parameters["remote_library_directory"],
        remote_model_paths_by_artifact={base["artifact_sha256"]: parameters["remote_model_path"]},
        remote_state_directory="/data/local/tmp/" + options.output.parent.name + "-" + options.output.name,
        executable_device_name="GPUOpenCL", output_directory=options.output,
    )
    launcher = AndroidLlamaServerProcessLauncher(config)
    sequence = 0

    def shell(command, required=True):
        nonlocal sequence
        started_ns = time.monotonic_ns()
        result = launcher._su(command, check=False, timeout_s=20)
        write_new(options.output / "commands" / f"{sequence:03d}.json", {
            "command": command, "started_ns": started_ns, "finished_ns": time.monotonic_ns(),
            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
        })
        sequence += 1
        if required and result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout

    stop = threading.Event()
    latest = []
    observer = None
    managed = None
    identity = None
    forwarded = False
    results = []
    pid_file = config.remote_state_directory + "/server.pid"
    remote_port, local_port = 18491, 29491
    launch_ns = ready_ns = None
    request_deadline_ns = None

    def check():
        if request_deadline_ns is not None and time.monotonic_ns() > request_deadline_ns:
            raise RuntimeError("Request exceeded the 180-second diagnostic bound")
        if managed is not None and managed.process.poll() is not None:
            raise RuntimeError("Phone server exited")
        if not latest:
            raise RuntimeError("No phone health observation")
        observation = latest[-1]
        value = observation.value
        if not observation.to_json()["valid"] or value is None:
            raise RuntimeError("Phone health is unavailable or stale: " + str(observation.failure_reason))
        if not value.thermal_qualified or value.temperature_millic >= 90000 or value.battery_ppm < 50000:
            raise RuntimeError("Phone health limits exceeded")
        if value.available_bytes < 805306368:
            raise RuntimeError("Phone memory reserve breached")

    def observe():
        with (options.output / "TELEMETRY.jsonl").open("x") as stream:
            while not stop.is_set():
                observed = probe_android_phone_runtime(options.serial, 5037, diagnostic=True)
                latest[:] = [observed]
                record = observed.to_json()
                if observed.value is not None:
                    record["values"] = asdict(observed.value)
                try:
                    clocks = launcher._su(
                        "cat /sys/class/kgsl/kgsl-3d0/gpuclk /sys/class/kgsl/kgsl-3d0/gpubusy",
                        timeout_s=2,
                    )
                    record["kgsl_gpuclk_gpubusy"] = clocks.stdout
                except Exception as error:
                    record["clock_error"] = repr(error)
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                stop.wait(2)

    try:
        processes = shell("ps -A -o PID,NAME; getprop sys.usb.config; uname -a; cat /proc/meminfo")
        if any(line.split()[-1] in {"llama-server", "llama-ffn-split-worker", "llama-ffn-split-resident-workers"}
               for line in processes.splitlines() if line.split()):
            raise RuntimeError("Another native inference process is active; it was not touched")
        hashes = shell("sha256sum " + shlex.quote(config.remote_server_path) + " "
                       + shlex.quote(parameters["remote_model_path"]) + " "
                       + shlex.quote(config.remote_library_directory) + "/*.so")
        hash_by_path = {line.split()[1]: "sha256:" + line.split()[0] for line in hashes.splitlines()}
        assert hash_by_path[config.remote_server_path] == parameters["remote_server_sha256"]
        assert hash_by_path[parameters["remote_model_path"]] == base["artifact_sha256"]
        write_new(options.output / "BINARY_HASHES.json", hash_by_path)
        shell("cd " + shlex.quote(config.remote_library_directory)
              + " && LD_LIBRARY_PATH=. " + shlex.quote(config.remote_server_path) + " --list-devices")
        shell("env | grep -E '^(GGML|LLAMA|S41)'", required=False)
        latest[:] = [probe_android_phone_runtime(options.serial, 5037, diagnostic=True)]
        check()
        if latest[-1].value.available_bytes < 3_000_000_000 + 805306368:
            raise RuntimeError("Declared whole-model peak plus reserve does not fit live RAM")
        shell("mkdir " + shlex.quote(config.remote_state_directory))
        shell("awk '$2 ~ /:483B$/ && $4 == \"0A\" {busy=1} END {exit busy ? 1 : 0}' "
              "/proc/net/tcp /proc/net/tcp6")
        launcher._adb("forward", "--no-rebind", "tcp:" + str(local_port), "tcp:" + str(remote_port))
        forwarded = True
        arguments = [config.remote_server_path, "--model", parameters["remote_model_path"],
                     "--alias", parameters["model_alias"], "--ctx-size", str(parameters["context_size"]),
                     "--parallel", str(parameters["parallel"]), "--batch-size", str(parameters["batch_size"]),
                     "--ubatch-size", str(parameters["ubatch_size"]), "--cont-batching",
                     "--cache-type-k", "f16", "--cache-type-v", "f16", "--host", "127.0.0.1",
                     "--port", str(remote_port), "--n-gpu-layers", str(options.gpu_layers),
                     "--device", "GPUOpenCL", "--flash-attn", options.flash_attn,
                     "--slots", "--metrics", "--no-webui", "--log-colors", "off", "--log-verbosity", "4"]
        body = ("cd " + shlex.quote(config.remote_library_directory) + " && export LD_LIBRARY_PATH=."
                + " && echo $$ > " + shlex.quote(pid_file) + " && exec " + shlex.join(arguments))
        command = tuple(launcher._adb_command(options.serial, "shell", "su -c " + shlex.quote(body)))
        contract = LlamaServerLaunchContract(
            model_alias=parameters["model_alias"], context_size=parameters["context_size"],
            parallel=parameters["parallel"], batch_size=parameters["batch_size"], ubatch_size=parameters["ubatch_size"],
            gpu_layers=options.gpu_layers, cpu_device_id="phone-cpu", gpu_device_id="op15-phone",
            phone_device_id=None, ffn_environment={},
        )
        write_new(options.output / "LAUNCH.json", {
            "command": command, "native_arguments": arguments,
            "contract": {field.name: dict(contract.ffn_environment) if field.name == "ffn_environment"
                         else getattr(contract, field.name) for field in fields(contract)},
            "scope": "standalone runtime diagnostic; no route or energy qualification",
            "differences_from_previous_probe": ["no HTP residency", "USB ADB instead of FunctionFS NCM",
                                                 "no scheduler monitor or memory-peak sampler", "log verbosity 4",
                                                 "no separate one-prompt/two-output-token warmup request"],
            "source_files": {str(path.relative_to(options.repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (options.repo / "research_dev/scheduler/adapters").glob("*.py")},
        })
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        managed = ManagedLlamaServer(command, os.environ.copy(), options.output, "phone", contract)
        launch_ns = time.monotonic_ns()
        managed.start()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            check()
            pid = launcher._su("cat " + shlex.quote(pid_file), check=False).stdout.strip()
            if pid.isdecimal():
                identity = parse_android_process_identity(
                    launcher._su(launcher._process_identity_command(int(pid), pid_file)).stdout, int(pid))
                if identity.executable != config.remote_server_path:
                    raise RuntimeError("Loaded executable identity differs")
                try:
                    if launcher._healthy(local_port):
                        ready_ns = time.monotonic_ns()
                        break
                except (OSError, ValueError):
                    pass
            time.sleep(0.25)
        if ready_ns is None:
            raise RuntimeError("Phone startup exceeded 120 seconds")
        write_new(options.output / "READY.json", {"launch_ns": launch_ns, "ready_ns": ready_ns,
                  "startup_s": (ready_ns - launch_ns) / 1e9, "process_identity": identity.to_json()})
        shell("cat /proc/" + str(identity.process_id) + "/maps")
        print("READY", (ready_ns - launch_ns) / 1e9, flush=True)
        for index in range(options.repetitions):
            first_tokens = []
            payload = LlamaCppCompletionPayload(
                request_id="llama-speed-" + str(index), expected_model_alias=parameters["model_alias"],
                input_tokens=request["input_tokens"], output_tokens=request["output_tokens"],
                prompt_tokens=tuple(request["prompt_tokens"]), seed=42,
                stream_path=options.output / f"request-{index}.raw", on_first_token=first_tokens.append,
                quality_mode="semantic", timeout_s=180,
            )
            started_ns = time.monotonic_ns()
            request_deadline_ns = started_ns + 180_000_000_000
            response = LlamaCppHttpClient().complete("http://127.0.0.1:" + str(local_port), payload, check)
            finished_ns = time.monotonic_ns()
            request_deadline_ns = None
            result = {"index": index, "started_ns": started_ns, "finished_ns": finished_ns,
                      "wall_s": (finished_ns - started_ns) / 1e9,
                      "ttft_s": (first_tokens[0] - started_ns) / 1e9,
                      "decode_tokens_per_s": request["output_tokens"] * 1000 / response["predicted_ms"],
                      "execution": response, "process_identity": identity.to_json()}
            write_new(options.output / f"EXECUTION_{index}.json", result)
            results.append(result)
            print("COMPLETED", index, result["wall_s"], result["decode_tokens_per_s"], flush=True)
        shell("cat /proc/" + str(identity.process_id) + "/status; dumpsys thermalservice")
        write_new(options.output / "RESULT.json", {"status": "PASS", "executions": results,
                  "startup_s": (ready_ns - launch_ns) / 1e9, "model_load_count": 1,
                  "energy_qualified": False, "scheduler_qualification": False})
    except BaseException as error:
        write_new(options.output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        stop.set()
        if observer is not None:
            observer.join(timeout=5)
        cleanup_error = None
        try:
            if identity is not None:
                launcher.stop_remote(identity.process_id, pid_file, expected=identity)
            if managed is not None:
                managed.stop()
            if forwarded:
                launcher.remove_forward(local_port)
        except BaseException as error:
            cleanup_error = repr(error)
        write_new(options.output / "CLEANUP.json", {"error": cleanup_error,
                  "status": "PASS" if cleanup_error is None else "FAIL",
                  "usb": shell("getprop sys.usb.config; ps -A -o PID,NAME", required=False)})
        if cleanup_error:
            raise RuntimeError(cleanup_error)


if __name__ == "__main__":
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("Diagnostic interrupted; cleaning up its own server")
    signal.signal(signal.SIGTERM, interrupted)
    main()
