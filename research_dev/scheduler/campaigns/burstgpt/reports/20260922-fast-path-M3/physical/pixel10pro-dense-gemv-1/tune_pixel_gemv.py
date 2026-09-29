"""Finite, unprofiled Pixel FFN sweep; launch only under the shared rig lock."""

import argparse
import json
from pathlib import Path
import shlex
import socket
import statistics
import time
from types import SimpleNamespace

import numpy as np

from profile_pixel_named import checked, run_arm
from qualify_op11_tcp import digest, save, worker_command


ORIGINAL = "/data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1"
PHONE_MODEL = "/data/local/tmp/s42-pixel10pro-qualification-20260922-v1/HTP0.ffn.gguf"
CPU_WORKER = "/mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin/llama-ffn-split-worker"
CPU_MODEL = "/home/zhihao/s42-op11-qwen-shards-20260921-v1/qwen/HTP0.ffn.gguf"
ORIGINAL_WORKER_SHA = "7cf01c7ae4940a92bd44d7d2b1b4a5bac0fe0c02d2953c8ef39158aa398d03ad"
ORIGINAL_VULKAN_SHA = "871e6b0733473edbfe339060c727fd58a89de001e25778aa58a3652fbae08e08"


def summarize(records):
    result = {}
    for columns in (8704, 17408):
        samples = [r for r in records if r["warm"] and r["columns"] == columns]
        result[str(columns)] = {"calls": len(samples)}
        for key in ("worker_us", "rpc_us"):
            values = [r[key] / 1000 for r in samples]
            result[str(columns)][key.removesuffix("_us")] = {
                "mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
                "p90_ms": float(np.percentile(values, 90)),
                "stdev_ms": statistics.stdev(values)}
    result["maximum_relative_l2"] = max(r["relative_l2"] for r in records)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--software", type=Path, required=True)
    parser.add_argument("--graph-software", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    arms = config["arms"]
    shader_variants = {}
    if any("shader" in arm for arm in arms):
        shader_variants = json.loads((args.software / "BUILD_PROVENANCE.json").read_text())["variants"]
    if len({a["name"] for a in arms}) != len(arms):
        raise ValueError("arm names must be unique")
    for arm in arms:
        if Path(arm["name"]).name != arm["name"] or arm["name"] in (".", ".."):
            raise ValueError("arm name must be a basename")
        if arm["quantum"] not in (4352, 8704) or arm["runtime"] not in ("original", "tuned"):
            raise ValueError("unsupported arm geometry or runtime")
        if arm.get("worker", "original") not in ("original", "reordered"):
            raise ValueError("unsupported worker")
        if arm.get("worker") == "reordered" and args.graph_software is None:
            raise ValueError("graph-reordering worker directory required")
        if "wg" in arm and (arm["wg"] not in (32, 64, 128, 256, 512) or arm["rows"] not in (1, 2, 4, 8)):
            raise ValueError("unsupported kernel specialization")
        if "subgroup" in arm and (arm["subgroup"] not in (32, 64, 128)
                                  or arm["wg"] < arm["subgroup"] or arm["wg"] % arm["subgroup"]):
            raise ValueError("unsupported subgroup specialization")
        if "shader" in arm and (arm["shader"] not in shader_variants or arm["runtime"] != "tuned"
                                or "wg" not in arm):
            raise ValueError("unsupported shader variant")
    if arms[0]["runtime"] != "original" or arms[0]["quantum"] != 4352:
        raise ValueError("first arm must be the qualified control")
    if not 4 <= config["repeats"] <= 20:
        raise ValueError("bounded repeat count required")
    phone_dir = config["phone_dir"]
    if not phone_dir.startswith("/data/local/tmp/s42-pixel10pro-gemv-tune-"):
        raise ValueError("private phone directory required")
    root = args.output
    root.mkdir()
    for name in (Path(__file__).name, "profile_pixel_named.py", "qualify_op11_tcp.py"):
        (root / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    save(root / "CONFIG.json", config)
    geometry = SimpleNamespace(artifact_sha256="sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718",
                               layers=list(range(18, 24)), n_embd=5120, columns=17408, quantum=4352)
    adb = ["adb", "-P", "5037", "-s", "5A040DLCH004ES"]
    cpu_port, phone_port = 27140, 27141
    forward = None
    try:
        host_processes = checked(["ps", "-eo", "pid,comm,args"])
        save(root / "HOST_PREFLIGHT.json", {"processes": host_processes})
        for line in host_processes.splitlines()[1:]:
            fields = line.split(maxsplit=2)
            if len(fields) > 1 and (fields[1] == "llama-server" or fields[1].startswith("llama-ffn")):
                raise RuntimeError("existing rig model process after lock acquisition; leave untouched")
        processes = checked(adb + ["shell", "ps -A -o PID,ARGS"])
        tcp = checked(adb + ["shell", "cat /proc/net/tcp /proc/net/tcp6"])
        boot = checked(adb + ["shell", "cat /proc/sys/kernel/random/boot_id"])
        save(root / "PREFLIGHT.json", {"processes": processes, "tcp": tcp, "boot_id": boot,
                                       "devices": checked(["adb", "-P", "5037", "devices", "-l"])})
        if "llama-ffn" in processes:
            raise RuntimeError("existing Pixel FFN worker; leave untouched")
        for line in tcp.splitlines():
            fields = line.split()
            if len(fields) > 3 and fields[3] == "0A" and int(fields[1].split(":")[-1], 16) == phone_port:
                raise RuntimeError("tuning phone port already listening")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", cpu_port))
        checked(adb + ["shell", "mkdir", phone_dir])
        checked(adb + ["push", str(args.software / "libggml-vulkan.so"), phone_dir + "/libggml-vulkan.so"], 60)
        if args.graph_software is not None:
            checked(adb + ["push", str(args.graph_software / "llama-ffn-split-worker"),
                           phone_dir + "/llama-ffn-split-worker"], 60)
            checked(adb + ["shell", "chmod", "755", phone_dir + "/llama-ffn-split-worker"])
        files = [PHONE_MODEL, ORIGINAL + "/llama-ffn-split-worker", phone_dir + "/libggml-vulkan.so"]
        if args.graph_software is not None:
            files.append(phone_dir + "/llama-ffn-split-worker")
        files += [ORIGINAL + "/" + name for name in (
            "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-vulkan.so")]
        hashes = checked(adb + ["shell", shlex.join(["sha256sum", *files])], 180)
        hash_map = {line.split()[1]: line.split()[0] for line in hashes.splitlines()}
        if hash_map[ORIGINAL + "/llama-ffn-split-worker"] != ORIGINAL_WORKER_SHA:
            raise RuntimeError("qualified worker bytes changed")
        if hash_map[ORIGINAL + "/libggml-vulkan.so"] != ORIGINAL_VULKAN_SHA:
            raise RuntimeError("qualified Vulkan library changed")
        if hash_map[phone_dir + "/libggml-vulkan.so"] != digest(args.software / "libggml-vulkan.so"):
            raise RuntimeError("tuning library hash differs")
        if args.graph_software is not None and hash_map[phone_dir + "/llama-ffn-split-worker"] != digest(
                args.graph_software / "llama-ffn-split-worker"):
            raise RuntimeError("graph-reordering worker hash differs")
        if hash_map[PHONE_MODEL] != digest(CPU_MODEL):
            raise RuntimeError("CPU and Pixel shard bytes differ")
        save(root / "IDENTITY.json", {"phone_hashes": hash_map, "boot_id": boot,
                                      "cpu_worker_sha256": digest(CPU_WORKER), "geometry": vars(geometry),
                                      "cpu_libraries": {p.name: digest(p) for p in Path(CPU_WORKER).parent.glob("libggml*.so")},
                                      "harness_sha256": digest(__file__)})
        cases = []
        inputs = {}
        for layer in geometry.layers:
            inputs[layer] = np.random.default_rng(20260922 + layer).normal(size=5120).astype("<f2").tobytes()
            (root / f"input-layer{layer}.f16").write_bytes(inputs[layer])
        for repeat in range(config["repeats"]):
            for layer in geometry.layers:
                for columns in ((8704, 17408) if repeat % 2 == 0 else (17408, 8704)):
                    cases.append((layer, columns, repeat, inputs[layer]))
        save(root / "CASES.json", [{"id": i, "layer": layer, "columns": columns, "repeat": repeat}
                                  for i, (layer, columns, repeat, _) in enumerate(cases, 1)])
        cpu_command = worker_command(geometry, CPU_WORKER, CPU_MODEL, "CPU", cpu_port, len(cases))
        _, references = run_arm(root / "cpu", cpu_command, cpu_port, geometry, cases)
        baseline_outputs = None
        summaries = {}
        for arm in arms:
            name = arm["name"]
            geometry.quantum = arm["quantum"]
            forward = checked(adb + ["forward", "--no-rebind", "tcp:0", f"tcp:{phone_port}"])
            runtime = ORIGINAL if arm["runtime"] == "original" else phone_dir + ":" + ORIGINAL
            command = ["env", "-u", "GGML_VK_PERF_LOGGER", "-u", "GGML_VK_DISABLE_FUSION",
                       "-u", "S42_PIXEL_PROFILE_OPS", "LD_LIBRARY_PATH=" + runtime]
            if "wg" in arm:
                command += [f"S42_PIXEL_F16_WG={arm['wg']}", f"S42_PIXEL_F16_ROWS={arm['rows']}"]
            if "subgroup" in arm:
                command += [f"S42_PIXEL_F16_SUBGROUP={arm['subgroup']}"]
            if "shader" in arm:
                command += [f"S42_PIXEL_F16_SHADER={arm['shader']}"]
            worker_dir = phone_dir if arm.get("worker") == "reordered" else ORIGINAL
            command += worker_command(geometry, worker_dir + "/llama-ffn-split-worker", PHONE_MODEL,
                                      "Vulkan0", phone_port, len(cases))
            command = adb + ["shell", "-T", "exec " + shlex.join(command)]
            battery_before = checked(adb + ["shell", "dumpsys battery"])
            records, outputs = run_arm(root / name, command, int(forward), geometry, cases, references)
            checked(adb + ["forward", "--remove", "tcp:" + forward])
            forward = None
            save(root / name / "THERMAL.json", {"battery_before": battery_before,
                                                "battery_after": checked(adb + ["shell", "dumpsys battery"])})
            log = (root / name / "worker.log").read_text()
            if "Timings:" in log or "PIXEL_FFN_STAGE" in log:
                raise RuntimeError("unexpected profiling in timing arm")
            selected = f"S42PIXELGEMV wg={arm.get('wg')} rows={arm.get('rows')} subgroup={arm.get('subgroup', 128)} cols=1"
            if "wg" in arm and selected not in log:
                raise RuntimeError("requested kernel specialization was not selected")
            if "shader" in arm and f"S42PIXELSHADER name={arm['shader']} precision=f32 cols=1" not in log:
                raise RuntimeError("requested shader variant was not selected")
            if baseline_outputs is None:
                baseline_outputs = outputs
            exact = sum(a == b for a, b in zip(outputs, baseline_outputs))
            if arm.get("require_exact", False) and exact != len(cases):
                raise RuntimeError("control outputs changed")
            summary = summarize(records)
            summary.update(arm=arm, exact_calls_vs_control=exact, calls=len(cases))
            summaries[name] = summary
            save(root / name / "SUMMARY.json", summary)
            with (root / "SUMMARY.jsonl").open("a") as journal:
                journal.write(json.dumps({"name": name, **summary}) + "\n")
            print(name, "SUMMARY", json.dumps(summary), flush=True)
        final_processes = checked(adb + ["shell", "ps -A -o PID,ARGS"])
        final_boot = checked(adb + ["shell", "cat /proc/sys/kernel/random/boot_id"])
        if final_boot != boot or "llama-ffn" in final_processes:
            raise RuntimeError("phone boot/process cleanup differs")
        save(root / "CLEANUP.json", {"status": "PASS", "boot_id": final_boot, "processes": final_processes,
                                     "forwards": checked(adb + ["forward", "--list"])})
        save(root / "RESULT.json", {"status": "PASS", "calls_per_arm": len(cases),
                                    "phone_calls": len(cases) * len(arms), "arms": summaries,
                                    "energy_measured": False, "full_model_tokens_verified": False,
                                    "finished_epoch_s": time.time()})
    except BaseException as error:
        save(root / "FAILURE.json", {"error": repr(error), "retained_forward": forward,
                                     "workers_not_killed": True, "finished_epoch_s": time.time()})
        raise


if __name__ == "__main__":
    main()
