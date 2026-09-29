"""Measure an exact CPU parent with the canonical launcher, not a route policy."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def sha(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "server", "cuda-libs", "model", "request", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from research_dev.scheduler import GGUFModelManifestLoader
    from research_dev.scheduler.adapters import (
        HostEnergySampler, LlamaCppCompletionPayload, LlamaCppHttpClient,
        LlamaServerLaunchContract, LlamaServerProcessConfiguration, LlamaServerProcessLauncher,
        default_host_metric_callbacks, nvidia_gpu_snapshot, probe_nvidia_process_memory_bytes,
        server_energy_summary,
    )
    from research_dev.scheduler.campaigns.burstgpt.desktop_parent_calibration import _endpoint_is_free

    args.output.mkdir()
    endpoint = "http://127.0.0.1:18684"
    if not _endpoint_is_free(endpoint):
        raise RuntimeError("calibration endpoint is occupied")
    source = json.loads(args.request.read_text())
    model_id = "llama-3.2-1b-instruct-q4_0"
    manifest = GGUFModelManifestLoader.load(model_id, args.model)
    if manifest.artifact_sha256 != "sha256:" + source["model_artifact_sha256"]:
        raise RuntimeError("calibration request artifact differs")
    parameters = dict(model_alias=model_id, context_size=4096, parallel=1,
                      batch_size=1024, ubatch_size=256, gpu_layers=0,
                      threads=4, threads_batch=8, cuda_graph_mode="default",
                      desktop_launch_mode="canonical", cpu_device_id="desktop-cpu",
                      gpu_device_id="desktop-cuda")
    contract = LlamaServerLaunchContract(**parameters, phone_device_id=None, ffn_environment={})
    launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(
        server_path=args.server, model_paths_by_artifact={manifest.artifact_sha256: args.model},
        library_paths_by_device={"desktop-cuda": (args.cuda_libs,)},
        executable_device_names={"desktop-cuda": "CUDA0"},
        output_directory=args.output, common_library_paths=(args.server.parent, args.cuda_libs),
    ))
    sampler = HostEnergySampler(default_host_metric_callbacks(), interval_s=0.1)
    process = None
    stop = threading.Event()
    memory = []
    runs = []
    result = None
    sampler.start()
    try:
        deadline = time.monotonic() + 10
        while len(sampler.rows()) < 2:
            if time.monotonic() > deadline:
                raise RuntimeError("host sampler is unavailable")
            time.sleep(0.05)
        started = time.monotonic_ns()
        process = launcher.launch_contract(endpoint, contract, manifest, label="cpu-parent",
                                           control_check=lambda: None)
        ready = time.monotonic_ns()

        def observe():
            while not stop.wait(0.1):
                status = Path("/proc") / str(process.pid) / "status"
                if status.exists():
                    fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                              for line in status.read_text().splitlines()}
                    memory.append({"observed_ns": time.monotonic_ns(),
                                   "rss_bytes": int(fields["VmRSS"].split()[0]) * 1024,
                                   "high_water_bytes": int(fields["VmHWM"].split()[0]) * 1024})

        thread = threading.Thread(target=observe, daemon=True)
        thread.start()
        for label in ("cold", "hot"):
            first = []
            payload = LlamaCppCompletionPayload(
                request_id="cpu-parent-" + label, expected_model_alias=model_id,
                input_tokens=int(source["input_tokens"]), output_tokens=int(source["output_tokens"]),
                prompt_tokens=tuple(source["prompt_tokens"]), seed=42,
                stream_path=args.output / (label + ".raw"), on_first_token=first.append,
                quality_mode="semantic", timeout_s=120,
            )
            begin = time.monotonic_ns()
            response = LlamaCppHttpClient().complete(endpoint, payload, lambda: None)
            end = time.monotonic_ns()
            runs.append(dict(label=label, started_ns=begin, finished_ns=end, first_token_ns=first[0],
                             duration_us=(end - begin) // 1000, response=response))
        time.sleep(0.2)
        for row in runs:
            row["energy"] = server_energy_summary(
                sampler.rows_between(row["started_ns"], row["finished_ns"]),
                row["started_ns"], row["finished_ns"])
        result = dict(status="PASS", artifact_sha256=manifest.artifact_sha256,
                      runtime_parameters=parameters, model_manifest=manifest.to_json(),
                      server_sha256=sha(args.server), libraries={p.name: sha(p) for p in args.server.parent.glob("*.so")},
                      request=source, command=list(process.command), runs=runs,
                      load=dict(start_ns=started, ready_ns=ready, duration_us=(ready-started)//1000,
                                energy=server_energy_summary(sampler.rows_between(started, ready), started, ready)),
                      memory=memory, gpu_process_bytes=probe_nvidia_process_memory_bytes(process.pid),
                      gpu=nvidia_gpu_snapshot(), calibration_context="isolated CPU parent, not concurrent-work qualification")
    except BaseException as error:
        write(args.output / "FAILURE.json", dict(error=repr(error), traceback=traceback.format_exc(), runs=runs))
        raise
    finally:
        stop.set()
        if process is not None:
            process.stop()
        sampler.stop()
        write(args.output / "POWER_SAMPLES.json", sampler.rows())
        write(args.output / "POWER_DIAGNOSTICS.json", sampler.diagnostics())
    write(args.output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "runs_us": [r["duration_us"] for r in runs]}), flush=True)


if __name__ == "__main__":
    main()
