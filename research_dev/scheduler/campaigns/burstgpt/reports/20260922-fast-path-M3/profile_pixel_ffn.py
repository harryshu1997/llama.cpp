"""Bounded Pixel FFN stage/operator profile; run under the shared rig lock."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import socket
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from qualify_op11_tcp import (
    EXEC_REQUEST, EXEC_RESPONSE, MAGIC, connect, digest, fnv, hello,
    receive, save, worker_command,
)


def checked(command, timeout=30):
    return subprocess.check_output(command, stdin=subprocess.DEVNULL,
                                   text=True, timeout=timeout).strip()


def run_arm(root, command, port, geometry, cases, references=None):
    root.mkdir()
    save(root / "COMMAND.json", command)
    records = []
    outputs = []
    with (root / "worker.log").open("x") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        save(root / "PROCESS.json", {"controller_pid": process.pid})
        try:
            deadline = time.monotonic() + 240
            while "[ffn-worker] ready backend=" not in (root / "worker.log").read_text():
                if process.poll() is not None:
                    raise RuntimeError(f"worker startup exit {process.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("worker readiness deadline")
                time.sleep(0.25)
            print(root.name, "READY", flush=True)
            with connect(port) as stream:
                save(root / "HELLO.json", hello(stream, geometry))
                for ident, (layer, columns, repeat, payload) in enumerate(cases, 1):
                    packet = EXEC_REQUEST.pack(MAGIC, 6, 3, ident, layer,
                                              geometry.n_embd, len(payload),
                                              fnv(payload), columns, 1) + payload
                    started = time.monotonic_ns()
                    stream.sendall(packet)
                    fields = EXEC_RESPONSE.unpack(receive(stream, EXEC_RESPONSE.size))
                    expected = (MAGIC, 6, 4, 0, 0, ident, layer, geometry.n_embd, len(payload))
                    if fields[:9] != expected or fields[10:12] != (columns, 1):
                        raise RuntimeError(f"response header mismatch: {fields!r}")
                    output = receive(stream, len(payload))
                    rpc_us = (time.monotonic_ns() - started) / 1000
                    if fnv(output) != fields[9]:
                        raise RuntimeError("response payload hash mismatch")
                    values = np.frombuffer(output, dtype="<f2").astype(np.float64)
                    if not np.isfinite(values).all():
                        raise RuntimeError("nonfinite output")
                    (root / f"output-{ident:03d}.f16").write_bytes(output)
                    outputs.append(output)
                    row = {"id": ident, "layer": layer, "columns": columns,
                           "repeat": repeat, "warm": repeat >= 2,
                           "rpc_us": rpc_us, "worker_us": fields[12],
                           "input_sha256": hashlib.sha256(payload).hexdigest(),
                           "output_sha256": hashlib.sha256(output).hexdigest()}
                    if references is not None:
                        reference = np.frombuffer(references[ident - 1], dtype="<f2").astype(np.float64)
                        row["relative_l2"] = float(np.linalg.norm(values - reference) /
                                                   max(np.linalg.norm(reference), 1e-30))
                    records.append(row)
                    with (root / "CALLS.jsonl").open("a") as journal:
                        journal.write(json.dumps(row) + "\n")
            status = process.wait(timeout=30)
            if status != 0:
                raise RuntimeError(f"worker exit {status}")
            failed = [r for r in records if r.get("relative_l2", 0) > 0.01]
            save(root / "RESULT.json", {"status": "FAIL" if failed else "PASS",
                                       "calls": len(records), "exit_code": status,
                                       "maximum_relative_l2": max(r.get("relative_l2", 0) for r in records),
                                       "normal_finite_exit": True})
            if failed:
                raise RuntimeError(f"{len(failed)} numerical comparisons failed")
            print(root.name, "PASS", len(records), flush=True)
            return records, outputs
        except BaseException as error:
            save(root / "FAILURE.json", {"error": repr(error), "calls": len(records),
                                        "process_status": process.poll(), "worker_not_killed": True})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--software", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir()
    (root / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    geometry = SimpleNamespace(artifact_sha256="sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718",
                               layers=list(range(18, 24)), n_embd=5120, columns=17408, quantum=4352)
    adb = ["adb", "-P", "5037", "-s", "5A040DLCH004ES"]
    original = "/data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1"
    phone_dir = "/data/local/tmp/s42-pixel10pro-profile-20260922-v1"
    phone_model = "/data/local/tmp/s42-pixel10pro-qualification-20260922-v1/HTP0.ffn.gguf"
    cpu_worker = "/mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin/llama-ffn-split-worker"
    cpu_model = "/home/zhihao/s42-op11-qwen-shards-20260921-v1/qwen/HTP0.ffn.gguf"
    forward = None
    try:
        processes = checked(adb + ["shell", "ps -A -o PID,ARGS"])
        tcp = checked(adb + ["shell", "cat /proc/net/tcp /proc/net/tcp6"])
        boot = checked(adb + ["shell", "cat /proc/sys/kernel/random/boot_id"])
        save(root / "PREFLIGHT.json", {"processes": processes, "tcp": tcp, "boot_id": boot,
                                       "devices": checked(["adb", "-P", "5037", "devices", "-l"])})
        if "llama-ffn" in processes:
            raise RuntimeError("existing Pixel FFN worker; leave untouched")
        ports = list(range(26991, 26995))
        for line in tcp.splitlines():
            fields = line.split()
            if len(fields) > 3 and fields[3] == "0A" and int(fields[1].split(":")[-1], 16) in ports:
                raise RuntimeError("profile phone port already listening")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 26990))
        checked(adb + ["shell", "mkdir", phone_dir])
        checked(adb + ["push", str(args.software / "llama-ffn-split-worker"), phone_dir + "/llama-ffn-split-worker"], 60)
        checked(adb + ["shell", "chmod", "755", phone_dir + "/llama-ffn-split-worker"])
        libraries = [original + "/" + name for name in (
            "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-vulkan.so")]
        phone_files = [phone_dir + "/llama-ffn-split-worker", original + "/llama-ffn-split-worker", phone_model, *libraries]
        hashes = checked(adb + ["shell", shlex.join(["sha256sum", *phone_files])], 180)
        hash_map = {line.split()[1]: line.split()[0] for line in hashes.splitlines()}
        if hash_map[phone_dir + "/llama-ffn-split-worker"] != digest(args.software / "llama-ffn-split-worker"):
            raise RuntimeError("instrumented worker hash differs")
        if hash_map[phone_model] != digest(cpu_model):
            raise RuntimeError("CPU and phone shard hashes differ")
        save(root / "IDENTITY.json", {"phone_hashes": hash_map, "boot_id": boot,
                                      "cpu_worker_sha256": digest(cpu_worker),
                                      "geometry": vars(geometry), "harness_sha256": digest(__file__)})
        cases = []
        inputs = {}
        for layer in geometry.layers:
            inputs[layer] = np.random.default_rng(20260922 + layer).normal(size=5120).astype("<f2").tobytes()
            (root / f"input-layer{layer}.f16").write_bytes(inputs[layer])
        for repeat in range(8):
            for layer in geometry.layers:
                for columns in ((8704, 17408) if repeat % 2 == 0 else (17408, 8704)):
                    cases.append((layer, columns, repeat, inputs[layer]))
        save(root / "CASES.json", [{"id": i, "layer": layer, "columns": columns, "repeat": repeat}
                                  for i, (layer, columns, repeat, _) in enumerate(cases, 1)])
        cpu_command = worker_command(geometry, cpu_worker, cpu_model, "CPU", 26990, len(cases))
        _, references = run_arm(root / "cpu", cpu_command, 26990, geometry, cases)
        baseline_outputs = None
        for index, name in enumerate(("control-before", "stage-timers", "gpu-profile", "control-after")):
            forward = checked(adb + ["forward", "--no-rebind", "tcp:0", f"tcp:{ports[index]}"])
            base = original if name.startswith("control") else phone_dir
            command = ["env", "LD_LIBRARY_PATH=" + original]
            if name == "gpu-profile":
                command += ["GGML_VK_PERF_LOGGER=1", "GGML_VK_PERF_LOGGER_FREQUENCY=1"]
            command += worker_command(geometry, base + "/llama-ffn-split-worker", phone_model,
                                      "Vulkan0", ports[index], len(cases))
            command = adb + ["shell", "-T", "exec " + shlex.join(command)]
            records, outputs = run_arm(root / name, command, int(forward), geometry, cases, references)
            checked(adb + ["forward", "--remove", "tcp:" + forward])
            forward = None
            if baseline_outputs is None:
                baseline_outputs = outputs
            if outputs != baseline_outputs:
                raise RuntimeError(f"{name} outputs differ from uninstrumented worker")
        final_processes = checked(adb + ["shell", "ps -A -o PID,ARGS"])
        final_boot = checked(adb + ["shell", "cat /proc/sys/kernel/random/boot_id"])
        if final_boot != boot or "llama-ffn" in final_processes:
            raise RuntimeError("phone boot/process cleanup differs")
        save(root / "CLEANUP.json", {"status": "PASS", "boot_id": final_boot,
                                     "processes": final_processes,
                                     "forwards": checked(adb + ["forward", "--list"])})
        save(root / "RESULT.json", {"status": "PASS", "calls_per_arm": len(cases),
                                    "phone_arms": 4, "all_phone_outputs_identical": True,
                                    "finished_epoch_s": time.time()})
    except BaseException as error:
        save(root / "FAILURE.json", {"error": repr(error), "retained_forward": forward,
                                     "workers_not_killed": True, "finished_epoch_s": time.time()})
        raise


if __name__ == "__main__":
    main()
