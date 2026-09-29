"""Run the saved Llama request once on an otherwise idle desktop CPU."""

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-dir", required=True)
    args = parser.parse_args()
    args.output.mkdir()
    rows = [json.loads(line) for line in args.requests.read_text().splitlines()]
    selected = [row for row in rows if row["combined_request_index"] == 37]
    assert len(selected) == 1
    row = selected[0]
    assert len(row["prompt_tokens"]) == row["input_tokens"] == 915
    assert row["output_tokens"] == 292
    assert digest(args.model) == row["model_artifact_sha256"]
    running = subprocess.check_output(["ps", "-eo", "comm="], text=True).splitlines()
    assert not any(name.strip() in ("llama-server", "llama-cli", "llama-bench") for name in running)
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    command = [
        str(args.server), "--model", str(args.model), "--alias", row["model_id"],
        "--device", "none", "--n-gpu-layers", "0", "--no-kv-offload",
        "--fit", "off", "--ctx-size", "2048", "--parallel", "1",
        "--batch-size", "2048", "--ubatch-size", "512",
        "--threads", "8", "--threads-batch", "8",
        "--host", "127.0.0.1", "--port", str(port), "--no-webui",
        "--log-colors", "off", "--log-timestamps",
    ]
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("LLAMA_ARG_", "S41_SERVER_FFN_", "LLAMA_FFN_SPLIT_"))
    }
    environment.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    environment["LD_LIBRARY_PATH"] = args.library_dir + ":" + str(args.server.parent)
    body = dict(cache_prompt=False, ignore_eos=True, n_predict=292,
                prompt=row["prompt_tokens"], return_tokens=True,
                seed=42, stream=True, temperature=0.0)
    save(args.output / "REQUEST.json", body)
    save(args.output / "SOURCE_REQUEST.json", row)
    save(args.output / "RUN_COMMAND.json", {
        "argv": command, "LD_LIBRARY_PATH": environment["LD_LIBRARY_PATH"],
        "server_sha256": digest(args.server), "model_sha256": digest(args.model),
        "probe_sha256": digest(Path(__file__)), "host": socket.gethostname(),
        "libraries": {p.name: digest(p) for p in args.server.parent.glob("*.so")},
        "request_sha256": digest(args.output / "REQUEST.json"),
    })
    process = None
    connection = None
    try:
        with (args.output / "server.stdout").open("x") as stdout, (args.output / "server.stderr").open("x") as stderr:
            launched = time.perf_counter()
            process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr)
            while True:
                if process.poll() is not None:
                    raise RuntimeError("CPU server exited during startup")
                if time.perf_counter() - launched > 90:
                    raise TimeoutError("CPU server startup exceeded 90 seconds")
                health = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                try:
                    health.request("GET", "/health")
                    response = health.getresponse()
                    healthy = response.status == 200
                    response.read()
                    if healthy:
                        break
                except OSError:
                    pass
                finally:
                    health.close()
                time.sleep(0.1)
            load_s = time.perf_counter() - launched
            print("CPU_READY", round(load_s, 3), "seconds; submitting exactly one full request", flush=True)
            rapl = Path("/sys/class/powercap/intel-rapl:0")
            before_energy = int((rapl / "energy_uj").read_text())
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
            started = time.perf_counter()
            connection.request("POST", "/completion", body=json.dumps(body),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(response.read().decode(errors="replace"))
            first_token_s = None
            final = None
            output = []
            with (args.output / "completion.raw").open("xb") as stream:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    stream.write(line)
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    item = json.loads(line[6:])
                    output.append(item.get("content", ""))
                    if first_token_s is None and (item.get("tokens") or item.get("content")):
                        first_token_s = time.perf_counter() - started
                    if item.get("stop"):
                        final = item
            request_s = time.perf_counter() - started
            after_energy = int((rapl / "energy_uj").read_text())
            assert final is not None
            timings = final["timings"]
            assert timings["prompt_n"] == 915 and timings["predicted_n"] == 292
            assert timings["cache_n"] == 0
            energy_uj = (after_energy - before_energy) % int((rapl / "max_energy_range_uj").read_text())
            save(args.output / "FINAL_RESPONSE.json", final)
            with (args.output / "output.txt").open("x") as stream:
                stream.write("".join(output))
            result = {
                "status": "PASS", "scope": "one CPU-only request; not an energy comparison",
                "model": row["model_id"], "threads": 8, "gpu_layers": 0,
                "input_tokens": 915, "output_tokens": 292, "seed": 42,
                "startup_to_ready_s": load_s, "request_wall_s": request_s,
                "time_to_first_token_s": first_token_s, "timings": timings,
                "cpu_package_energy_j": energy_uj / 1e6,
                "raw_sha256": digest(args.output / "completion.raw"),
            }
            save(args.output / "RESULT.json", result)
            print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        save(args.output / "FAILURE.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        if connection is not None:
            connection.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        save(args.output / "CLEANUP.json", {
            "owned_server_pid": None if process is None else process.pid,
            "owned_server_exit_code": None if process is None else process.poll(),
            "unrelated_processes_changed": False,
        })


if __name__ == "__main__":
    main()
