"""Bounded Pixel Vulkan FFN server comparison; run under the shared rig lock."""

import argparse
import http.client
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.environ["S42_UNIFIED_REPO_ROOT"])

from research_dev.scheduler._internal.adaptive_decode_contracts import (  # noqa: E402
    AdaptiveDecodeControl, AdaptiveDecodePolicy,
)
from research_dev.scheduler._internal.types import canonical_sha256  # noqa: E402
from research_dev.scheduler.adapters import (  # noqa: E402
    HostEnergySampler, LlamaCppHttpClient, default_host_metric_callbacks,
)
from research_dev.scheduler.adapters.http_backend import LlamaCppCompletionPayload  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.remote_resident_gate import _energy  # noqa: E402
from qualify_op11_tcp import (  # noqa: E402
    EXEC_REQUEST, EXEC_RESPONSE, MAGIC, connect, digest, fnv, hello, receive,
    save, worker_command,
)


def request_json(port, method, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, None if body is None else json.dumps(body),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {path}: {response.status}: {data[:200]!r}")
        return json.loads(data)
    finally:
        connection.close()


def wait_ready(process, check, seconds=240):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"startup exit {process.returncode}")
        try:
            if check():
                return
        except (OSError, RuntimeError):
            pass
        time.sleep(0.25)
    raise TimeoutError("readiness deadline exceeded; process retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.output
    root.mkdir()
    save(root / "CONFIG.json", config)
    adb = ["adb", "-P", "5037", "-s", config["serial"]]
    phone = SimpleNamespace(**config["phone"])
    phone_limit = config["output_tokens"] * 2
    server = worker = None
    forward = None
    sampler = HostEnergySampler(default_host_metric_callbacks(), interval_s=0.1)
    records = []
    try:
        for port in (config["server_port"],):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
        processes = subprocess.check_output(adb + ["shell", "ps -A -o PID,ARGS"],
                                            stdin=subprocess.DEVNULL, text=True, timeout=20)
        if "llama-ffn-split" in processes:
            raise RuntimeError("Pixel has an existing FFN worker")
        tcp = subprocess.check_output(adb + ["shell", "cat /proc/net/tcp /proc/net/tcp6"],
                                     stdin=subprocess.DEVNULL, text=True, timeout=20)
        if any(line.split()[1].endswith(f":{config['phone_port']:04X}")
               for line in tcp.splitlines() if len(line.split()) > 3):
            raise RuntimeError("Pixel worker port is occupied")
        boot = subprocess.check_output(adb + ["shell", "cat /proc/sys/kernel/random/boot_id"],
                                      stdin=subprocess.DEVNULL, text=True, timeout=20).strip()
        files = [config["phone_worker"], *config["phone_libraries"]]
        hashes = subprocess.check_output(adb + ["shell", shlex.join(["sha256sum", *files])],
                                        stdin=subprocess.DEVNULL, text=True, timeout=60)
        identity = {"boot_id": boot, "phone_hashes": hashes,
                    "server_sha256": digest(config["server"]), "harness_sha256": digest(__file__),
                    "server_libraries": {str(p): digest(p) for p in Path(config["server"]).parent.glob("*.so")},
                    "parent_stat": list(Path(config["model"]).stat()),
                    "parent_sha256_from_previous_verification": phone.artifact_sha256}
        save(root / "IDENTITY.json", identity)
        worker_argv = worker_command(phone, config["phone_worker"], config["phone_model"],
                                     "Vulkan0", config["phone_port"], phone_limit)
        worker_argv = adb + ["shell", "-T", "exec " + shlex.join(
            ["env", "LD_LIBRARY_PATH=" + config["phone_library_dir"], *worker_argv])]
        save(root / "WORKER_COMMAND.json", worker_argv)
        worker_log = (root / "worker.log").open("x")
        worker = subprocess.Popen(worker_argv, stdin=subprocess.DEVNULL,
                                  stdout=worker_log, stderr=subprocess.STDOUT)
        wait_ready(worker, lambda: "[ffn-worker] ready backend=" in (root / "worker.log").read_text())
        forward = subprocess.check_output(adb + ["forward", "--no-rebind", "tcp:0",
                                                f"tcp:{config['phone_port']}"], text=True).strip()
        save(root / "FORWARD.json", {"host_port": int(forward), "phone_port": config["phone_port"]})
        env = {k: v for k, v in os.environ.items() if not k.startswith("S41_SERVER_FFN_")}
        env.update(config["server_environment"])
        env["S41_SERVER_FFN_PORT"] = forward
        command = [config["server"], *config["server_arguments"]]
        save(root / "SERVER_COMMAND.json", {"argv": command, "environment": config["server_environment"],
                                            "S41_SERVER_FFN_PORT": forward})
        server_log = (root / "server.log").open("x")
        server = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                  stdout=server_log, stderr=subprocess.STDOUT)
        wait_ready(server, lambda: request_json(config["server_port"], "GET", "/health"))
        listeners = subprocess.check_output(["ss", "-ltnp", f"sport = :{config['server_port']}"], text=True)
        if f"pid={server.pid}," not in listeners:
            raise RuntimeError("answering server PID differs")
        save(root / "SERVER_READY.json", {"pid": server.pid, "listeners": listeners})
        print("SERVER_READY", server.pid, flush=True)
        tokens = request_json(config["server_port"], "POST", "/tokenize",
                      {"content": config["prompt_text"], "add_special": True})["tokens"]
        tokens = tuple(tokens[:config["prompt_tokens"]])
        if len(tokens) != config["prompt_tokens"]:
            raise RuntimeError("prompt is too short")
        save(root / "PROMPT.json", list(tokens))
        endpoint = f"http://127.0.0.1:{config['server_port']}"
        client = LlamaCppHttpClient()
        sampler.start()
        wait_ready(server, lambda: len(sampler.rows()) >= 4, 30)
        for index, columns in enumerate((0, 8704, 17408, 0)):
            rid = f"pixel-ffn-{index}-columns{columns}"
            first, progress, controls = [], [], []

            def on_first(t_ns):
                first.append(t_ns)
                if columns:
                    policy = AdaptiveDecodePolicy(
                        route_id=rid, executor_id="pixel-vulkan-qualification",
                        operator_plan_sha256=canonical_sha256({"config": config, "columns": columns}),
                        desktop_parent_route_id="pixel-qualification-parent",
                        desktop_placement_sha256=canonical_sha256(command),
                        layer_indices=tuple(phone.layers), layer_mask=sum(1 << x for x in phone.layers),
                        columns=columns, split_fraction_ppm=columns * 1000000 // phone.columns,
                        resource_ids=("desktop-cpu", "desktop-cuda", "pixel-vulkan"))
                    control = AdaptiveDecodeControl(rid, 0, index + 1, policy)
                    ack, received = client.apply_ffn_control(endpoint, control, timeout_s=180)
                    controls.append({"ack": ack, "received_ns": received, "control": control.to_json()})

            payload = LlamaCppCompletionPayload(
                request_id=rid, expected_model_alias=config["alias"], input_tokens=len(tokens),
                output_tokens=config["output_tokens"], prompt_tokens=tokens, seed=17,
                stream_path=root / f"{rid}.raw", on_first_token=on_first,
                on_decode_progress=lambda *values: progress.append(values),
                quality_mode="accounting-only", timeout_s=600)
            started = time.monotonic_ns()
            result = client.complete(endpoint, payload, lambda: None)
            finished = time.monotonic_ns()
            if len(result["tokens"]) != config["output_tokens"] or not first:
                raise RuntimeError("incomplete generated tokens")
            row = {"request_id": rid, "columns": columns, "started_ns": started,
                   "first_token_ns": first[0], "finished_ns": finished,
                   "request_s": (finished - started) / 1e9,
                   "decode_s": (finished - first[0]) / 1e9,
                   "request_host_energy": _energy(sampler, started, finished),
                   "decode_host_energy": _energy(sampler, first[0], finished),
                   "controls": controls, "progress": progress, "execution": result}
            save(root / f"REQUEST-{index}.json", row)
            records.append(row)
            print("REQUEST_DONE", index, columns, row["request_s"], flush=True)
        if any(s.get("is_processing") for s in client.slots("127.0.0.1", config["server_port"], 10)):
            raise RuntimeError("server remains active after completed requests")
        server.send_signal(signal.SIGINT)
        server_exit = server.wait(timeout=60)
        if server_exit != 0:
            raise RuntimeError(f"server shutdown status {server_exit}")
        proof_lines = [line for line in (root / "server.log").read_text().splitlines()
                       if "S41SERVERFFNCALL " in line]
        calls = len(proof_lines)
        if not 0 < calls < phone_limit:
            raise RuntimeError(f"unexpected server call count {calls}")
        save(root / "SERVER_CALLS.json", proof_lines)
        cleanup_calls = []
        with connect(int(forward)) as stream:
            hello(stream, phone)
            data = np.zeros(phone.n_embd, dtype="<f2").tobytes()
            for ident in range(1, phone_limit - calls + 1):
                stream.sendall(EXEC_REQUEST.pack(MAGIC, 6, 3, ident, phone.layers[0],
                                                phone.n_embd, len(data), fnv(data), phone.columns, 1) + data)
                fields = EXEC_RESPONSE.unpack(receive(stream, EXEC_RESPONSE.size))
                output = receive(stream, len(data))
                if fields[:9] != (MAGIC, 6, 4, 0, 0, ident, phone.layers[0], phone.n_embd, len(data)):
                    raise RuntimeError("cleanup response differs")
                if fields[9] != fnv(output) or fields[10:12] != (phone.columns, 1):
                    raise RuntimeError("cleanup response hash/shape differs")
                cleanup_calls.append(ident)
        worker_exit = worker.wait(timeout=30)
        if worker_exit != 0:
            raise RuntimeError(f"worker shutdown status {worker_exit}")
        subprocess.run(adb + ["forward", "--remove", "tcp:" + forward], check=True, timeout=20)
        forward = None
        boot_after = subprocess.check_output(adb + ["shell", "cat /proc/sys/kernel/random/boot_id"],
                                            stdin=subprocess.DEVNULL, text=True, timeout=20).strip()
        if boot_after != boot:
            raise RuntimeError("phone rebooted")
        expected = records[0]["execution"]["tokens"]
        matches = [r["execution"]["tokens"] == expected for r in records]
        summary = {"status": "PASS" if all(matches) else "FAIL", "token_identity": matches,
                   "output_tokens_each": len(expected), "phone_calls": calls,
                   "call_columns": {str(c): sum(bool(re.search(rf"\bcolumns={c}\b", line)) for line in proof_lines)
                                    for c in (8704, 17408)},
                   "server_exit": server_exit, "worker_exit": worker_exit,
                   "cleanup_calls_outside_measurement": len(cleanup_calls),
                   "phone_energy_measured": False, "scheduler_integrated": False,
                   "requests": [{k: v for k, v in r.items() if k not in ("execution", "progress")}
                                for r in records], "finished_epoch_s": time.time()}
        save(root / "RESULT.json", summary)
        return 0 if all(matches) else 1
    except BaseException as error:
        save(root / "FAILURE.json", {"error": repr(error), "retained_forward": forward,
                                     "worker_status": worker.poll() if worker else None,
                                     "server_status": server.poll() if server else None,
                                     "processes_not_killed": True, "finished_epoch_s": time.time()})
        raise
    finally:
        sampler.stop()
        save(root / "POWER_SAMPLES.json", sampler.rows())
        save(root / "POWER_DIAGNOSTICS.json", sampler.diagnostics())


if __name__ == "__main__":
    raise SystemExit(main())
