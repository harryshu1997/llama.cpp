"""Finite compiled TPU probes over ADB5037, holding the Pixel device lock."""

import argparse
import hashlib
import json
import re
import shlex
import statistics
import struct
import subprocess
import time

import numpy as np
from pathlib import Path
from qualify_op11_tcp import connect, digest, receive, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phone-dir", required=True)
    parser.add_argument("--serial", default="5A040DLCH004ES")
    parser.add_argument("--port", type=int, default=26971)
    parser.add_argument("--model", default="ffn.tflite")
    parser.add_argument("--vectors", type=Path)
    parser.add_argument("--calls", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.calls <= 1000:
        parser.error("require 0 <= warmup < calls <= 1000")
    args.output.mkdir()
    vectors = np.load(args.vectors) if args.vectors else None
    case_count = len(vectors["inputs"]) if vectors is not None else 10
    numerical_pass = True
    adb = ["adb", "-P", "5037", "-s", args.serial]
    forward = None
    process = None

    def shell(command):
        return subprocess.check_output(adb + ["shell", command], stdin=subprocess.DEVNULL,
                                       text=True, timeout=30).strip()

    try:
        processes = shell("ps -A -o PID,ARGS")
        if any(name in processes for name in ("litert_tpu_probe", "litert_ffn_probe", "llama-ffn-split", "pixel-bandwidth")):
            raise RuntimeError("phone already has a qualification worker")
        tcp = shell("cat /proc/net/tcp /proc/net/tcp6")
        if any(line.split()[1].endswith(f":{args.port:04X}")
               for line in tcp.splitlines() if len(line.split()) > 3):
            raise RuntimeError("phone port already in use")
        boot = shell("cat /proc/sys/kernel/random/boot_id")
        files = [args.phone_dir + "/" + name for name in (
            "litert_ffn_probe", "libLiteRt.so", "libLiteRtDispatch_GoogleTensor.so",
            args.model)]
        save(args.output / "IDENTITY.json", {
            "serial": args.serial, "boot_id": boot, "model": shell("getprop ro.product.model"),
            "hashes": shell(shlex.join(["sha256sum", *files])),
            "harness_sha256": digest(__file__), "helper_sha256": digest(Path(__file__).with_name("qualify_op11_tcp.py"))})
        forward = subprocess.check_output(adb + ["forward", "--no-rebind", "tcp:0", f"tcp:{args.port}"],
                                          stdin=subprocess.DEVNULL, text=True).strip()
        save(args.output / "FORWARD.json", {"host_port": int(forward), "phone_port": args.port})
        command = ["env", "LD_LIBRARY_PATH=" + args.phone_dir + ":/vendor/lib64", files[0], files[3],
                   args.phone_dir, str(args.port), str(args.calls)]
        save(args.output / "COMMAND.json", command)
        log_path = args.output / "worker.log"
        records = []
        with log_path.open("x") as log:
            locked = "flock -n 9 9>&9 || exit 73; exec " + shlex.join(command) + " 9>&9"
            shell_command = "sh -c " + shlex.quote(locked) + " 9>/data/local/tmp/.s42-pixel-ffn-kernels.lock"
            save(args.output / "LOCKED_COMMAND.json", shell_command)
            process = subprocess.Popen(adb + ["shell", shell_command], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 120
            ready = None
            while ready is None:
                ready = re.search(r"ready port=\d+ input_bytes=(\d+) output_bytes=(\d+)", log_path.read_text())
                if ready:
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"worker startup exit {process.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("worker readiness")
                time.sleep(0.1)
            input_bytes, output_bytes = map(int, ready.groups())
            if input_bytes != (1 if vectors is not None else 2) * output_bytes or output_bytes % 4:
                raise RuntimeError("unexpected probe geometry")
            count = output_bytes // 4
            print(f"ready input_bytes={input_bytes} output_bytes={output_bytes}", flush=True)
            with connect(int(forward)) as stream:
                for i in range(args.calls):
                    rng = np.random.default_rng(20260922 + i % 10)
                    values = (rng.integers(-16, 17, size=2 * count).astype("<f4") / 8).astype("<f4")
                    if vectors is not None:
                        values = vectors["inputs"][i % case_count].astype("<f4").reshape(-1)
                    payload = values.tobytes()
                    if len(payload) != input_bytes:
                        raise RuntimeError("input shape mismatch")
                    started = time.monotonic_ns()
                    stream.sendall(struct.pack("<II", i, input_bytes) + payload)
                    ident, invoke_ns, worker_ns = struct.unpack("<QQQ", receive(stream, 24))
                    output = receive(stream, output_bytes)
                    rpc_ns = time.monotonic_ns() - started
                    actual = np.frombuffer(output, dtype="<f4")
                    if ident != i:
                        raise RuntimeError("response ID mismatch")
                    expected = (vectors["reference"][i % case_count].reshape(-1) if vectors is not None
                                else values[:count] + values[count:])
                    relative_l2 = float(np.linalg.norm(actual.astype(np.float64)-expected) / np.linalg.norm(expected))
                    exact = bool(np.array_equal(actual, expected))
                    numerical_pass &= bool(np.isfinite(actual).all() and (relative_l2 <= 0.01 if vectors is not None else exact))
                    (args.output / f"input-{i:03d}.f32").write_bytes(payload)
                    (args.output / f"output-{i:03d}.f32").write_bytes(output)
                    record = {"id": i, "warmup": i < args.warmup, "rpc_ns": rpc_ns,
                              "invoke_ns": invoke_ns, "worker_ns": worker_ns, "case": i % case_count,
                              "relative_l2": relative_l2, "reference_exact": exact,
                              "input_sha256": hashlib.sha256(payload).hexdigest(),
                              "output_sha256": hashlib.sha256(output).hexdigest()}
                    records.append(record)
                    with (args.output / "CALLS.jsonl").open("a") as journal:
                        journal.write(json.dumps(record) + "\n")
            status = process.wait(timeout=30)
            save(args.output / "EXIT.json", {"status": status, "calls": len(records)})
            if status != 0:
                raise RuntimeError(f"worker exit {status}")
        subprocess.run(adb + ["forward", "--remove", "tcp:" + forward],
                       stdin=subprocess.DEVNULL, check=True, timeout=20)
        forward = None
        if shell("cat /proc/sys/kernel/random/boot_id") != boot:
            raise RuntimeError("phone rebooted")
        worker_log = log_path.read_text()
        proof = {
            "npu_only": "accelerator=npu_only" in worker_log,
            "google_dispatch": "libLiteRtDispatch_GoogleTensor.so" in worker_log,
            "vendor_runtime": "SouthBound symbols resolved by 'libedgetpu_litert.so'" in worker_log,
            "entire_graph_delegated": any(a == b and int(a) > 0 for a, b in re.findall(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(DispatchDelegate\)", worker_log)),
            "normal_exit": f"PASS calls={args.calls} normal_exit=1" in worker_log,
        }
        warm = records[args.warmup:]
        timings = {}
        fields = {"invoke": lambda r: r["invoke_ns"], "worker": lambda r: r["worker_ns"],
                  "round_trip": lambda r: r["rpc_ns"],
                  "outside_worker": lambda r: r["rpc_ns"] - r["worker_ns"]}
        for name, field in fields.items():
            values = [field(r) / 1e6 for r in warm]
            timings[name] = {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
                             "p90_ms": float(np.percentile(values, 90)), "min_ms": min(values), "max_ms": max(values)}
        repeats_exact = all(len({r["output_sha256"] for r in records if r["case"] == n}) <= 1
                            for n in range(case_count))
        result = {"status": "PASS" if all(proof.values()) and repeats_exact and numerical_pass else "FAIL",
                  "model": args.model,
                  "calls": len(records), "warm_calls": len(warm), "input_bytes": input_bytes,
                  "output_bytes": output_bytes, "numerical_pass": numerical_pass, "relative_l2_limit": 0.01 if vectors is not None else 0.0,
                  "max_relative_l2": max(r["relative_l2"] for r in records),
                  "numerical_exact": all(r["reference_exact"] for r in records),
                  "load_ms": int(re.search(r"load_ns=(\d+)", worker_log).group(1))/1e6, "repeats_exact": repeats_exact,
                  "tpu_proof": proof, "timing": timings, "finished_epoch_s": time.time(),
                  "qwen_ffn_tested": vectors is not None, "energy_measured": False, "scheduler_integrated": False}
        save(args.output / "RESULT.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["status"] == "PASS" else 1
    except BaseException as error:
        if forward and (process is None or process.poll() is not None):
            subprocess.run(adb + ["forward", "--remove", "tcp:" + forward], stdin=subprocess.DEVNULL, timeout=20, check=False)
            forward = None
        save(args.output / "FAILURE.json", {"error": repr(error), "retained_forward": forward,
                                           "process_status": process.poll() if process else None,
                                           "workers_not_killed": True, "finished_epoch_s": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
