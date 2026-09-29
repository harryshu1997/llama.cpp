"""Single-request split-KV hardware gate. No phone and no scheduler qualification claim."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from research_dev.scheduler.adapters import HostEnergySampler, default_host_metric_callbacks
from research_dev.scheduler.adapters.http_backend import LlamaCppCompletionPayload, LlamaCppHttpClient
from research_dev.scheduler.campaigns.burstgpt.remote_resident_gate import _energy


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def memory(pid):
    result = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "VmSwap"):
            result[key] = int(value.split()[0]) * 1024
    return result


def runtime_libraries(pid):
    paths = {Path(line.split()[-1]) for line in Path(f"/proc/{pid}/maps").read_text().splitlines()
             if "/libllama" in line or "/libggml" in line}
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def request(port, method, path, payload=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(method, path, None if payload is None else json.dumps(payload), {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {data}")
        return data
    finally:
        conn.close()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.reference_config.read_text())
    document = Path(config["prompt_file"]).read_text() + config.get("prompt_suffix", "")
    # This Qwen fixture has 40 decoder layers; the native GPU-layer count includes the output layer.
    gpu_decoder_layers = range(40 + 1 - 16, 40)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(args.server), "--model", config["model"], "--alias", "split-kv-test", "--fit", "off",
               "--ctx-size", "32768", "--parallel", "1", "--batch-size", "512", "--ubatch-size", "128",
               "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16", "--kv-unified",
               "--n-gpu-layers", "16", "--split-mode", "none", "--threads", "8", "--threads-batch", "8",
               "--host", "127.0.0.1", "--port", str(port), "--slots", "--no-webui", "--no-warmup",
               "--no-cache-idle-slots", "--log-colors", "off"]
    if args.mode == "host":
        command += ["--kv-cpu-layers", ",".join(map(str, gpu_decoder_layers))]
    elif args.mode == "split":
        command += ["--kv-device-cells", ",".join(f"{il}:{args.device_cells}" for il in gpu_decoder_layers)]
    # Preserve other users' processes and settings; this child has no FFN owner or helper environment.
    env = {key: value for key, value in os.environ.items() if not key.startswith("S41_SERVER_FFN_")}
    env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    env.pop("LLAMA_KV_CACHE_EAGER_CLEAR", None)
    sampler = HostEnergySampler(default_host_metric_callbacks(), interval_s=0.1)
    process = None
    result = {"mode": args.mode, "command": command, "scheduler_qualified": False,
              "phone_used": False, "single_run": True, "status": "RUNNING",
              "server_sha256": hashlib.sha256(args.server.read_bytes()).hexdigest()}
    save(args.output / "CONFIG.json", {**vars(args), "server": str(args.server), "output": str(args.output),
                                       "reference_config": str(args.reference_config), "lock": str(args.lock)})
    try:
        sampler.start()
        deadline = time.monotonic() + 20
        while len(sampler.rows()) < 2:
            if time.monotonic() > deadline:
                raise RuntimeError("host energy samples unavailable")
            time.sleep(0.1)
        loaded_start = time.monotonic_ns()
        with (args.output / "server.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        deadline = time.monotonic() + 240
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"server exited {process.returncode}; see server.log")
            try:
                request(port, "GET", "/health")
                break
            except (OSError, RuntimeError):
                if time.monotonic() > deadline:
                    raise RuntimeError("server preparation timed out")
                time.sleep(0.2)
        result["load_s"] = (time.monotonic_ns() - loaded_start) / 1e9
        props = request(port, "GET", "/props")
        save(args.output / "PROPS.json", props)
        expected_prefixes = ([{"layer": il, "device_cells": args.device_cells} for il in gpu_decoder_layers]
                             if args.mode == "split" else [])
        expected_host = list(gpu_decoder_layers) if args.mode == "host" else []
        if props.get("kv_device_cells") != expected_prefixes or props.get("kv_cpu_layers") != expected_host:
            raise RuntimeError("server KV placement differs from the requested gate arm")
        save(args.output / "RUNTIME_LIBRARIES.json", runtime_libraries(process.pid))
        result["memory_before_request"] = memory(process.pid)
        tokens = request(port, "POST", "/tokenize", {"content": document, "add_special": True})["tokens"]
        if len(tokens) != 9737:
            raise RuntimeError(f"expected the established 9737-token prompt, got {len(tokens)}")
        save(args.output / "REQUEST.json", {"tokens": tokens, "output_tokens": args.tokens,
            "prompt_sha256": hashlib.sha256(document.encode()).hexdigest()})
        first = []
        def first_token(stamp):
            first.append(stamp)
            result["memory_first_token"] = memory(process.pid)
            print(f"FIRST_TOKEN mode={args.mode} t={time.monotonic():.3f}", flush=True)
        payload = LlamaCppCompletionPayload("split-kv-test", "split-kv-test", len(tokens), args.tokens, tuple(tokens),
                                            17, args.output / "STREAM.jsonl", first_token, timeout_s=1200)
        started = time.monotonic_ns()
        print(f"REQUEST mode={args.mode} prompt={len(tokens)} output={args.tokens}", flush=True)
        response = LlamaCppHttpClient().complete(f"http://127.0.0.1:{port}", payload, lambda: None)
        finished = time.monotonic_ns()
        save(args.output / "RESPONSE.json", response)
        result.update(status="PASS", request_s=(finished - started)/1e9, prompt_tokens=len(tokens),
                      output_tokens=list(response.get("tokens", ())), prompt_ms=response.get("prompt_ms"),
                      predicted_ms=response.get("predicted_ms"), memory_finished=memory(process.pid))
        result["host_energy"] = _energy(sampler, started, finished)
        result["host_energy"]["boundary"] = "request_only_excluding_model_load"
        if first:
            result["prefill_s"] = (first[0] - started)/1e9
            result["decode_s"] = (finished - first[0])/1e9
            result["decode_host_energy"] = _energy(sampler, first[0], finished)
            result["decode_host_energy"]["boundary"] = "first_output_token_to_request_end"
        if len(result["output_tokens"]) != args.tokens:
            raise RuntimeError("output token count differs from the request")
    except Exception as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        sampler.stop()
        save(args.output / "POWER_SAMPLES.json", sampler.rows())
        save(args.output / "RESULT.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("host", "device", "split"), required=True)
    parser.add_argument("--device-cells", type=int, default=8192)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
