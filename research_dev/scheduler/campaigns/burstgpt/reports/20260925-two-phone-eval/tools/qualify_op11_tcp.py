"""Bounded protocol-v6 FFN check. Run under the rig lock; never kills workers."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import socket
import statistics
import struct
import subprocess
import time

import numpy as np


MAGIC = 0x46534631
HELLO_REQUEST = struct.Struct("<IHHQIIHH32s4x")
HELLO_RESPONSE = struct.Struct("<IHHHHIIIIII4xQQIHH32s")
EXEC_REQUEST = struct.Struct("<IHHIiIIIII")
EXEC_RESPONSE = struct.Struct("<IHHHHIiIIIIIQ")


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fnv(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value


def receive(stream, count):
    data = bytearray()
    while len(data) < count:
        part = stream.recv(count - len(data))
        if not part:
            raise RuntimeError("worker disconnected before complete response")
        data.extend(part)
    return bytes(data)


def connect(port):
    stream = socket.create_connection(("127.0.0.1", port), timeout=60)
    stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return stream


def hello(stream, args, *, wrong_artifact=False, missing_layer=False):
    artifact = bytes.fromhex(args.artifact_sha256.removeprefix("sha256:"))
    mask = sum(1 << layer for layer in args.layers)
    supplied_artifact = bytes([artifact[0] ^ 1]) + artifact[1:] if wrong_artifact else artifact
    supplied_mask = mask & ~(1 << args.layers[0]) if missing_layer else mask
    stream.sendall(HELLO_REQUEST.pack(MAGIC, 6, 1, supplied_mask, args.n_embd,
                                     args.columns, 3, 4, supplied_artifact))
    raw = receive(stream, HELLO_RESPONSE.size)
    fields = HELLO_RESPONSE.unpack(raw)
    expected = (MAGIC, 6, 2, int(wrong_artifact or missing_layer), 3,
                args.n_embd, args.columns, 0, args.columns,
                getattr(args, "weight_type", 1), len(args.layers), mask)
    if fields[:12] != expected or fields[13:] != (args.quantum, 4, 0, artifact):
        raise RuntimeError(f"HELLO identity/geometry/status mismatch: {fields!r}")
    if wrong_artifact or missing_layer:
        if stream.recv(1) != b"":
            raise RuntimeError("rejected HELLO did not close the connection")
    return {"raw_hex": raw.hex(), "weight_hash": f"{fields[12]:016x}",
            "artifact_sha256": artifact.hex(), "layer_mask": mask,
            "n_embd": fields[5], "columns": fields[8], "max_tokens": fields[14],
            "status": fields[3]}


def worker_command(args, worker, model, backend, port, count):
    return [worker, "-m", model, "--artifact-sha256", args.artifact_sha256,
            "--layers", ",".join(map(str, args.layers)), "--columns", str(args.columns),
            "--column-quantum", str(args.quantum), "--backend", backend,
            "--port", str(port), "--bind", "127.0.0.1", "--f16-io",
            "--max-tokens", "4", "--max-requests", str(count)]


def run_worker(args, label, command, port, cases):
    root = args.output / label
    root.mkdir()
    save(root / "COMMAND.json", command)
    records = []
    log_path = root / "worker.log"
    with log_path.open("x") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        save(root / "PROCESS.json", {"controller_pid": process.pid,
                                    "started_epoch_s": time.time()})
        try:
            deadline = time.monotonic() + 240
            while "[ffn-worker] ready backend=" not in log_path.read_text():
                if process.poll() is not None:
                    raise RuntimeError(f"{label} startup exit {process.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{label} worker readiness")
                time.sleep(0.25)
            print(label, "ready", flush=True)
            rejections = []
            for invalid in ("wrong_artifact", "missing_layer"):
                with connect(port) as stream:
                    rejections.append(hello(stream, args, **{invalid: True}))
            save(root / "REJECTIONS.json", rejections)
            with connect(port) as stream:
                identity = hello(stream, args)
                save(root / "HELLO.json", identity)
                for ident, (layer, rows, repeat, payload) in enumerate(cases, 1):
                    request = EXEC_REQUEST.pack(MAGIC, 6, 3, ident, layer,
                                                args.n_embd * rows, len(payload),
                                                fnv(payload), args.columns, rows)
                    started = time.monotonic_ns()
                    stream.sendall(request + payload)
                    fields = EXEC_RESPONSE.unpack(receive(stream, EXEC_RESPONSE.size))
                    expected = (MAGIC, 6, 4, 0, 0, ident, layer, args.n_embd * rows, len(payload))
                    if fields[:9] != expected or fields[10:12] != (args.columns, rows):
                        raise RuntimeError(f"execute header mismatch: {fields!r}")
                    output = receive(stream, len(payload))
                    rpc_us = (time.monotonic_ns() - started) / 1000
                    if fields[9] != fnv(output):
                        raise RuntimeError("execute payload hash mismatch")
                    if not np.isfinite(np.frombuffer(output, dtype="<f2")).all():
                        raise RuntimeError("nonfinite FFN output")
                    (root / f"output-{ident:03d}.f16").write_bytes(output)
                    record = {"id": ident, "layer": layer, "rows": rows, "repeat": repeat,
                              "rpc_us": rpc_us, "compute_us": fields[12],
                              "input_sha256": hashlib.sha256(payload).hexdigest(),
                              "output_sha256": hashlib.sha256(output).hexdigest()}
                    records.append(record)
                    with (root / "CALLS.jsonl").open("a") as journal:
                        journal.write(json.dumps(record) + "\n")
            status = process.wait(timeout=30)
            if status != 0:
                raise RuntimeError(f"{label} worker exit {status}")
            save(root / "EXIT.json", {"status": status, "normal_finite_request_exit": True,
                                      "calls": len(records), "finished_epoch_s": time.time()})
            return identity, records
        except BaseException as error:
            save(root / "FAILURE.json", {"error": repr(error), "completed_calls": len(records),
                                         "process_status": process.poll(),
                                         "worker_not_killed": True})
            raise


def compare(args, cases, reference, candidate):
    rows_checked = []
    for ident, (layer, rows, repeat, _) in enumerate(cases, 1):
        filename = f"output-{ident:03d}.f16"
        cpu = np.fromfile(args.output / "cpu" / filename, dtype="<f2").astype(np.float64)
        phone = np.fromfile(args.output / "candidate" / filename, dtype="<f2").astype(np.float64)
        for row in range(rows):
            start, stop = row * args.n_embd, (row + 1) * args.n_embd
            expected, actual = cpu[start:stop], phone[start:stop]
            squared_error = float(np.dot(actual - expected, actual - expected))
            squared_reference = float(np.dot(expected, expected))
            nmse = squared_error / max(squared_reference, 1e-30)
            rows_checked.append({"id": ident, "layer": layer, "rows": rows,
                                 "repeat": repeat, "row": row, "relative_l2": nmse ** 0.5,
                                 "nmse": nmse, "max_abs": float(np.max(np.abs(actual - expected))),
                                 "equal_element_fraction": float(np.mean(actual == expected))})
    save(args.output / "ROW_COMPARISON.json", rows_checked)
    timing = {}
    for name, records in (("cpu", reference), ("candidate", candidate)):
        timing[name] = {}
        for rows in (1, 2, 4):
            samples = [r for r in records if r["rows"] == rows and r["repeat"] > 0]
            timing[name][str(rows)] = {
                "calls": len(samples),
                "median_rpc_ms": statistics.median(r["rpc_us"] for r in samples) / 1000,
                "median_compute_ms": statistics.median(r["compute_us"] for r in samples) / 1000,
                "median_noncompute_ms": statistics.median(
                    r["rpc_us"] - r["compute_us"] for r in samples) / 1000,
                "maximum_rpc_ms": max(r["rpc_us"] for r in samples) / 1000}
    deterministic = all(len({r["output_sha256"] for r in candidate
                             if r["layer"] == layer and r["rows"] == rows}) == 1
                        for layer in args.layers for rows in (1, 2, 4))
    failures = [row for row in rows_checked if row["relative_l2"] > 0.01]
    return {"status": "FAIL" if failures else "PASS", "calls_per_worker": len(cases),
            "rows_checked": len(rows_checked), "relative_l2_limit": 0.01,
            "maximum_relative_l2": max(r["relative_l2"] for r in rows_checked),
            "maximum_nmse": max(r["nmse"] for r in rows_checked),
            "maximum_absolute_error": max(r["max_abs"] for r in rows_checked),
            "rows_above_limit": len(failures), "candidate_repeats_bit_exact": deterministic,
            "timing": timing, "energy_measured": False,
            "full_model_tokens_verified": False, "scheduler_integrated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--serial")
    parser.add_argument("--phone-label", default="OP11")
    parser.add_argument("--phone-dir", default="/data/local/tmp/s42-op11-20260921-bin")
    parser.add_argument("--phone-backend", choices=("GPUOpenCL", "HTP0", "Vulkan0", "CPU"), default="GPUOpenCL")
    parser.add_argument("--phone-library", action="append", help="Runtime library basename to hash; repeat for each library")
    parser.add_argument("--phone-nhmx", type=int, choices=(0, 1))
    parser.add_argument("--phone-model", default="/data/local/tmp/s42-op11-qwen-shards-20260921-v1/HTP0.ffn.gguf")
    parser.add_argument("--layers", type=int, nargs="+", default=[18, 19, 20, 21])
    parser.add_argument("--n-embd", type=int, default=5120)
    parser.add_argument("--columns", type=int, default=17408)
    parser.add_argument("--quantum", type=int, default=512)
    parser.add_argument("--cpu-port", type=int, default=26921)
    parser.add_argument("--candidate-port", type=int, default=26922)
    args = parser.parse_args()
    args.output.mkdir()
    save(args.output / "CONFIG.json", {**vars(args), "output": str(args.output),
                                      "harness_sha256": digest(__file__)})
    adb = ["adb", "-P", "5037", "-s", args.serial] if args.serial else None
    forwarded = None
    try:
        identity = {"host_shard_sha256": digest(args.model), "host_worker_sha256": digest(args.worker)}
        for port in (args.cpu_port, args.candidate_port) if not adb else (args.cpu_port,):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
        linked = subprocess.check_output(["ldd", args.worker], text=True, timeout=20)
        identity["host_linked_libraries"] = linked
        identity["host_library_hashes"] = {
            str(path): digest(path) for path in sorted(Path(args.worker).parent.glob("libggml*.so"))}
        if adb:
            processes = subprocess.check_output(adb + ["shell", "ps -A -o PID,ARGS"],
                                                text=True, timeout=20)
            save(args.output / "PHONE_PREFLIGHT.json", {"processes": processes})
            if "llama-ffn-split" in processes:
                raise RuntimeError(f"{args.phone_label} already has an FFN process; leave it untouched")
            tcp = subprocess.check_output(adb + ["shell", "cat /proc/net/tcp /proc/net/tcp6"],
                                         text=True, timeout=20)
            if any(line.split()[1].endswith(f":{args.candidate_port:04X}")
                   for line in tcp.splitlines() if len(line.split()) > 3):
                raise RuntimeError(f"{args.phone_label} qualification port is already in use")
            print(f"hashing {args.phone_label} shard and binaries", flush=True)
            libraries = args.phone_library or (
                "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so",
                "libggml-hexagon.so", "libggml-htp-v73.so", "libllama.so", "libomp.so")
            if any(Path(name).name != name for name in libraries):
                raise ValueError("phone libraries must be basenames")
            files = [args.phone_model, args.phone_dir + "/llama-ffn-split-worker",
                     *[args.phone_dir + "/" + name for name in libraries]]
            hashes = subprocess.check_output(adb + ["shell", shlex.join(["sha256sum", *files])],
                                             text=True, timeout=180)
            identity["phone_hashes"] = hashes
            if hashes.split()[0] != identity["host_shard_sha256"]:
                raise RuntimeError("phone/CPU shard bytes differ")
            for name, command in (("kernel", ["uname", "-a"]),
                                  ("boot_id", ["cat", "/proc/sys/kernel/random/boot_id"]),
                                  ("model", ["getprop", "ro.product.model"])):
                identity[name] = subprocess.check_output(adb + ["shell", shlex.join(command)],
                                                        text=True, timeout=20).strip()
        save(args.output / "IDENTITY.json", identity)
        cases = []
        for layer in args.layers:
            values = np.random.default_rng(20260922 + layer).normal(size=(4, args.n_embd)).astype("<f2")
            for rows in (1, 2, 4):
                payload = values[:rows].tobytes()
                (args.output / f"input-layer{layer}-rows{rows}.f16").write_bytes(payload)
                cases.extend((layer, rows, repeat, payload) for repeat in range(4))
        cpu_command = worker_command(args, args.worker, args.model, "CPU", args.cpu_port, len(cases))
        cpu_identity, reference = run_worker(args, "cpu", cpu_command, args.cpu_port, cases)
        if adb:
            forwarded = subprocess.check_output(adb + ["forward", "--no-rebind", "tcp:0",
                                                       f"tcp:{args.candidate_port}"], text=True).strip()
            save(args.output / "FORWARD.json", {"serial": args.serial, "adb_port": 5037,
                                               "host_port": int(forwarded),
                                               "phone_port": args.candidate_port})
            phone_command = worker_command(args, args.phone_dir + "/llama-ffn-split-worker",
                                           args.phone_model, args.phone_backend, args.candidate_port, len(cases))
            command = ["env", "LD_LIBRARY_PATH=" + args.phone_dir,
                       "ADSP_LIBRARY_PATH=" + args.phone_dir + ";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp"]
            if args.phone_nhmx is not None:
                command.append(f"GGML_HEXAGON_NHMX={args.phone_nhmx}")
            command.extend(phone_command)
            candidate_command = adb + ["shell", "-T", "echo QUALIFICATION_PID=$$; exec " + shlex.join(command)]
            port = int(forwarded)
        else:
            candidate_command = worker_command(args, args.worker, args.model, "CPU",
                                               args.candidate_port, len(cases))
            port = args.candidate_port
        test_identity, candidate = run_worker(args, "candidate", candidate_command, port, cases)
        if test_identity != cpu_identity:
            raise RuntimeError("CPU/candidate HELLO identities differ")
        if adb:
            subprocess.run(adb + ["forward", "--remove", "tcp:" + forwarded], check=True, timeout=20)
            forwarded = None
            boot_after = subprocess.check_output(adb + ["shell", "cat /proc/sys/kernel/random/boot_id"],
                                                 text=True, timeout=20).strip()
            if boot_after != identity["boot_id"]:
                raise RuntimeError("phone rebooted during qualification")
        result = compare(args, cases, reference, candidate)
        result["mode"] = f"{args.phone_label}-{args.phone_backend}-over-ADB-TCP" if adb else "local-CPU-harness-check"
        result["finished_epoch_s"] = time.time()
        save(args.output / "RESULT.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["status"] == "PASS" else 1
    except BaseException as error:
        save(args.output / "FAILURE.json", {"error": repr(error), "finished_epoch_s": time.time(),
                                           "retained_forward": forwarded, "workers_not_killed": True})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
