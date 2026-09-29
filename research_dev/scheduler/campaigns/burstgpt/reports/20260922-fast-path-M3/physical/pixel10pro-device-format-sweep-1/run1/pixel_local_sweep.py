"""Prepare and audit a phone-local finite sweep using archived CPU references."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
from types import SimpleNamespace

import numpy as np

from analyze_pixel_gemv import analyze
from qualify_op11_tcp import EXEC_REQUEST, EXEC_RESPONSE, HELLO_REQUEST, HELLO_RESPONSE, MAGIC, fnv, save, worker_command
from tune_pixel_gemv import ORIGINAL, ORIGINAL_VULKAN_SHA, ORIGINAL_WORKER_SHA, PHONE_MODEL, summarize


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(args):
    root, reference = args.root, args.reference
    config = read(args.config)
    config["phone_dir"] = args.phone_dir
    config["measurement_scope"] = "phone-local worker time; no host CPU run, energy or RPC measurement"
    warmup = config.get("warmup_repeats", 2)
    if type(warmup) is not int or not 0 <= warmup < config["repeats"]:
        raise ValueError("warmup repeats must leave measured calls")
    count = 12 * config["repeats"]
    identity = read(reference / "IDENTITY.json")
    geometry = SimpleNamespace(**identity["geometry"])
    cases = read(reference / "CASES.json")[:count]
    cpu_rows = [json.loads(line) for line in (reference / "cpu/CALLS.jsonl").read_text().splitlines()][:count]
    if read(reference / "RESULT.json")["status"] != "PASS" or len(cases) != count or len(cpu_rows) != count:
        raise ValueError("insufficient qualified reference cases")
    if not args.phone_dir.startswith("/data/local/tmp/s42-pixel10pro-gemv-tune-"):
        raise ValueError("private phone directory required")
    if any(arm["quantum"] != 4352 or arm.get("worker", "original") not in ("original", "preserved_input", "cpu_tuned", "cpu_gpu") for arm in config["arms"]):
        raise ValueError("unsupported worker or quantum")
    if any(arm["runtime"] not in ("original", "tuned", "previous") for arm in config["arms"]):
        raise ValueError("unsupported runtime")
    if any(arm.get("backend", "Vulkan0") not in ("Vulkan0", "CPU") for arm in config["arms"]):
        raise ValueError("unsupported backend")
    root.mkdir()
    (root / "cpu").mkdir()
    save(root / "CONFIG.json", config)
    save(root / "CASES.json", cases)
    save(root / "REFERENCE_IDENTITY.json", identity)
    inputs = {layer: (reference / f"input-layer{layer}.f16").read_bytes() for layer in geometry.layers}
    for layer, data in inputs.items():
        (root / f"input-layer{layer}.f16").write_bytes(data)
    artifact = bytes.fromhex(geometry.artifact_sha256.removeprefix("sha256:"))
    packet = bytearray(HELLO_REQUEST.pack(MAGIC, 6, 1, sum(1 << n for n in geometry.layers),
                                        geometry.n_embd, geometry.columns, 3, 4, artifact))
    hashes = {}
    for case, row in zip(cases, cpu_rows):
        ident, layer, columns = case["id"], case["layer"], case["columns"]
        if any(case[k] != row[k] for k in ("id", "layer", "columns", "repeat")):
            raise ValueError("reference case differs")
        payload = inputs[layer]
        output = (reference / "cpu" / f"output-{ident:03d}.f16").read_bytes()
        if sha(payload) != row["input_sha256"] or sha(output) != row["output_sha256"]:
            raise ValueError("reference hash differs")
        (root / "cpu" / f"output-{ident:03d}.f16").write_bytes(output)
        hashes[str(ident)] = sha(output)
        packet.extend(EXEC_REQUEST.pack(MAGIC, 6, 3, ident, layer, geometry.n_embd,
                                        len(payload), fnv(payload), columns, 1) + payload)
    (root / "REQUESTS.bin").write_bytes(packet)
    save(root / "REFERENCE_PROVENANCE.json", {"source": str(reference.resolve()), "calls": count,
         "input_sha256": {str(k): sha(v) for k, v in inputs.items()}, "output_sha256": hashes,
         "requests_sha256": sha(packet), "cpu_reexecuted": False})
    expected = dict(identity["phone_hashes"])
    expected = {p: h for p, h in expected.items() if p.startswith(ORIGINAL + "/") or p == PHONE_MODEL}
    expected[args.phone_dir + "/libggml-vulkan.so"] = sha((args.software / "libggml-vulkan.so").read_bytes())
    if any(arm.get("worker") in ("preserved_input", "cpu_tuned") for arm in config["arms"]):
        expected[args.phone_dir + "/llama-ffn-split-worker"] = sha((args.worker_software / "llama-ffn-split-worker").read_bytes())
    if any(arm.get("worker") == "cpu_gpu" for arm in config["arms"]):
        expected[args.phone_dir + "/llama-ffn-split-worker-dual"] = sha((args.dual_software / "llama-ffn-split-worker").read_bytes())
    if any(arm["runtime"] == "previous" for arm in config["arms"]):
        expected[args.phone_dir + "/previous/libggml-vulkan.so"] = sha((args.previous_software / "libggml-vulkan.so").read_bytes())
    if any(arm.get("cpu_affinity_fix") for arm in config["arms"]):
        expected[args.phone_dir + "/cpu-affinity/libggml-cpu.so"] = sha((args.cpu_software / "libggml-cpu.so").read_bytes())
    save(root / "EXPECTED_PHONE_HASHES.json", expected)
    (root / "EXPECTED_HASHES.sha256").write_text("".join(f"{value}  {path}\n" for path, value in expected.items()))
    header = """#!/system/bin/sh
cd PHONE_DIR || exit 2
exec 9>/data/local/tmp/.s42-pixel-ffn-kernels.lock
flock -n 9 9>&9 || exit 3
mkdir raw || exit 4
ps -A -o PID,ARGS > raw/PROCESSES_BEFORE.txt
if grep -q '[l]lama-ffn' raw/PROCESSES_BEFORE.txt; then echo 'FAIL existing worker'; exit 5; fi
cat /proc/sys/kernel/random/boot_id > raw/BOOT_BEFORE.txt
sha256sum HASH_FILES > raw/HASHES.txt || exit 6
cmp raw/HASHES.txt EXPECTED_HASHES.sha256 || exit 6
frequencies() {
    for frequency_file in /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq /sys/class/devfreq/*/cur_freq; do
        echo "$frequency_file"
        cat "$frequency_file"
    done
}
run_arm() {
    name="$1"
    shift
    mkdir "raw/$name" || return 7
    dumpsys battery > "raw/$name/BATTERY_BEFORE.txt"
    frequencies > "raw/$name/FREQUENCIES_BEFORE.txt" 2>&1
    "$@" > "raw/$name/worker.log" 2>&1 &
    worker_pid=$!
    echo "$worker_pid" > "raw/$name/PID.txt"
    ready=0
    for attempt in $(seq 1 240); do
        if grep -q '\\[ffn-worker\\] ready backend=' "raw/$name/worker.log"; then ready=1; break; fi
        if ! kill -0 "$worker_pid" 2>/dev/null; then echo "FAIL $name startup"; return 8; fi
        sleep 1
    done
    if [ "$ready" != 1 ]; then echo "FAIL $name readiness"; return 9; fi
    cat "/proc/$worker_pid/status" > "raw/$name/WORKER_STATUS.txt"
    for status_file in /proc/$worker_pid/task/*/status; do
        grep -E '^(Name|Pid|Cpus_allowed_list):' "$status_file"
    done > "raw/$name/THREAD_STATUS.txt"
    nc -n -w 5 -W 120 127.0.0.1 27141 < REQUESTS.bin > "raw/$name/RESPONSES.bin"
    network_status=$?
    echo "$network_status" > "raw/$name/NETCAT_EXIT.txt"
    if [ "$network_status" != 0 ]; then echo "FAIL $name capture"; return 10; fi
    wait "$worker_pid"
    worker_status=$?
    echo "$worker_status" > "raw/$name/WORKER_EXIT.txt"
    if [ "$worker_status" != 0 ]; then echo "FAIL $name worker"; return 11; fi
    dumpsys battery > "raw/$name/BATTERY_AFTER.txt"
    frequencies > "raw/$name/FREQUENCIES_AFTER.txt" 2>&1
    echo "PASS $name"
}
"""
    script = header.replace("PHONE_DIR", shlex.quote(args.phone_dir)).replace("HASH_FILES", shlex.join(expected))
    for arm in config["arms"]:
        runtime = ORIGINAL if arm["runtime"] == "original" else args.phone_dir + ":" + ORIGINAL
        if arm["runtime"] == "previous":
            runtime = args.phone_dir + "/previous:" + ORIGINAL
        if arm.get("cpu_affinity_fix"):
            if arm.get("backend") != "CPU" or arm.get("worker") not in ("cpu_tuned", "cpu_gpu"):
                raise ValueError("CPU affinity library requires private CPU worker")
            runtime = args.phone_dir + "/cpu-affinity:" + runtime
        command = ["env", "-u", "GGML_VK_PERF_LOGGER", "-u", "GGML_VK_DISABLE_FUSION", "-u", "S42_PIXEL_PROFILE_OPS",
                   "-u", "S42_PIXEL_F16_WG", "-u", "S42_PIXEL_F16_ROWS", "-u", "S42_PIXEL_F16_SUBGROUP",
                   "-u", "S42_PIXEL_F16_SHADER", "-u", "S42_PIXEL_FUSE_SWIGLU",
                   "-u", "S42_PIXEL_CPU_THREADS", "-u", "S42_PIXEL_CPU_MASK", "-u", "S42_PIXEL_CPU_POOL",
                   "-u", "S42_PIXEL_GPU_BLOCK_MASK", "-u", "S42_PIXEL_GPU_HOST_MASK", "-u", "S42_PIXEL_CPU_HALF_COLUMNS",
                   "-u", "S42_PIXEL_COALESCE_FULL", "-u", "S42_PIXEL_PACKED_WEIGHTS",
                   "-u", "S42_PIXEL_CPU_QUANT_RESIDUAL",
                   "-u", "S42_PIXEL_CPU_PAIR_BLOCKS",
                   "-u", "S42_PIXEL_NEON_ROWS", "-u", "S42_PIXEL_NEON_CHUNK",
                   "-u", "S42_PIXEL_NEON_UNROLL", "-u", "S42_PIXEL_NEON_PREFETCH",
                   "-u", "S42_PIXEL_NEON_FMLAL",
                   "-u", "S42_PIXEL_GPU_LAYOUT_ROWS", "-u", "S42_PIXEL_GPU_LAYOUT_CHUNK",
                   "-u", "S42_PIXEL_GPU_EXPAND_F16",
                   "-u", "S43_FFN_SECONDARY_BACKEND",
                   "-u", "S43_FFN_SECONDARY_FRACTION", "-u", "S43_FFN_SECONDARY_ALIGN",
                   "-u", "S43_FFN_SECONDARY_MAX_TOKENS", "-u", "S43_FFN_DUAL_LOG_PERIOD",
                   "LD_LIBRARY_PATH=" + runtime]
        if "cpu_threads" in arm:
            if arm.get("backend") != "CPU" or arm.get("worker") not in ("cpu_tuned", "cpu_gpu"):
                raise ValueError("CPU tuning requires the private CPU worker")
            command += [f"S42_PIXEL_CPU_THREADS={arm['cpu_threads']}",
                        f"S42_PIXEL_CPU_MASK={arm.get('cpu_mask', '0')}",
                        f"S42_PIXEL_CPU_POOL={int(arm.get('cpu_pool', False))}"]
        if "gpu_block_mask" in arm:
            mask = int(arm["gpu_block_mask"], 16)
            if (arm.get("backend") != "CPU" or arm.get("worker") != "cpu_gpu" or
                    not arm.get("cpu_pool") or arm["runtime"] not in ("tuned", "previous") or not 0 < mask < 15):
                raise ValueError("whole-block splitting requires tuned CPU/GPU runtimes")
            command += [f"S42_PIXEL_GPU_BLOCK_MASK={mask:x}", "S43_FFN_SECONDARY_BACKEND=Vulkan0",
                        "S43_FFN_SECONDARY_FRACTION=0", "S43_FFN_DUAL_LOG_PERIOD=1"]
        if "cpu_half_columns" in arm:
            columns = arm["cpu_half_columns"]
            if (arm.get("gpu_block_mask") != "6" or arm["runtime"] not in ("tuned", "previous") or
                    type(columns) is not int or not 64 <= columns <= 8640 or columns % 64):
                raise ValueError("fine ratios require mask6 and 64-aligned CPU half-columns")
            command.append(f"S42_PIXEL_CPU_HALF_COLUMNS={columns}")
        if arm.get("coalesce_full"):
            if arm.get("worker") != "cpu_gpu" or "cpu_half_columns" not in arm:
                raise ValueError("coalescing requires the fine split worker")
            command.append("S42_PIXEL_COALESCE_FULL=1")
        if arm.get("packed_weights"):
            if arm.get("worker") not in ("cpu_tuned", "cpu_gpu"):
                raise ValueError("packed experiment requires a private worker")
            command.append("S42_PIXEL_PACKED_WEIGHTS=1")
        if arm.get("quant_residual"):
            if not arm.get("packed_weights") or arm.get("backend") != "CPU":
                raise ValueError("residual correction requires packed CPU")
            command.append("S42_PIXEL_CPU_QUANT_RESIDUAL=1")
        if arm.get("cpu_pair_blocks"):
            if not arm.get("packed_weights") or arm.get("backend") != "CPU" or arm.get("gpu_block_mask"):
                raise ValueError("CPU block pairing requires packed CPU-only mode")
            command.append("S42_PIXEL_CPU_PAIR_BLOCKS=1")
        if "gpu_host_mask" in arm:
            if "gpu_block_mask" not in arm or not 0 < int(arm["gpu_host_mask"], 16) < 256:
                raise ValueError("GPU host affinity requires a dual arm and valid mask")
            command.append(f"S42_PIXEL_GPU_HOST_MASK={arm['gpu_host_mask']}")
        for key, variable in (("neon_rows", "NEON_ROWS"), ("neon_chunk", "NEON_CHUNK"),
                              ("neon_unroll", "NEON_UNROLL"), ("neon_prefetch", "NEON_PREFETCH"),
                              ("neon_fmlal", "NEON_FMLAL"),
                              ("gpu_layout_rows", "GPU_LAYOUT_ROWS"), ("gpu_layout_chunk", "GPU_LAYOUT_CHUNK")):
            if key in arm:
                if arm.get("worker") != "cpu_gpu" or type(arm[key]) is not int or not 0 <= arm[key] <= 4096:
                    raise ValueError("layout knobs require the private worker and bounded integers")
                command.append(f"S42_PIXEL_{variable}={arm[key]}")
        if arm.get("gpu_expand_f16"):
            if arm.get("worker") != "cpu_gpu" or not arm.get("packed_weights"):
                raise ValueError("device-specific formats require packed input and the private worker")
            command.append("S42_PIXEL_GPU_EXPAND_F16=1")
        for key, variable in (("wg", "WG"), ("rows", "ROWS"), ("subgroup", "SUBGROUP"), ("shader", "SHADER")):
            if key in arm:
                command.append(f"S42_PIXEL_F16_{variable}={arm[key]}")
        if arm.get("fuse_swiglu"):
            if arm["runtime"] != "tuned" or "shader" not in arm:
                raise ValueError("SwiGLU fusion requires a custom shader")
            command.append("S42_PIXEL_FUSE_SWIGLU=1")
        worker_dir = args.phone_dir if arm.get("worker") in ("preserved_input", "cpu_tuned") else ORIGINAL
        worker = (args.phone_dir + "/llama-ffn-split-worker-dual" if arm.get("worker") == "cpu_gpu"
                  else worker_dir + "/llama-ffn-split-worker")
        command += worker_command(geometry, worker, PHONE_MODEL,
                                  arm.get("backend", "Vulkan0"), 27141, count)
        script += shlex.join(["run_arm", arm["name"], *command]) + " || exit $?\n"
    script += """cat /proc/sys/kernel/random/boot_id > raw/BOOT_AFTER.txt
ps -A -o PID,ARGS > raw/PROCESSES_AFTER.txt
if grep -q '[l]lama-ffn' raw/PROCESSES_AFTER.txt; then echo 'FAIL remaining worker'; exit 12; fi
cmp raw/BOOT_BEFORE.txt raw/BOOT_AFTER.txt || exit 13
echo PASS > raw/DONE.txt
echo 'PASS all arms'
"""
    (root / "RUN_PHONE.sh").write_text(script)
    (root / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(f"PREPARED {count * len(config['arms'])} phone calls; {count} archived CPU references")


def audit(args):
    root = args.root
    raw = root / "raw"
    config, cases = read(root / "CONFIG.json"), read(root / "CASES.json")
    identity = read(root / "REFERENCE_IDENTITY.json")
    if (raw / "DONE.txt").read_text().strip() != "PASS":
        raise ValueError("phone sweep incomplete")
    if (raw / "BOOT_BEFORE.txt").read_bytes() != (raw / "BOOT_AFTER.txt").read_bytes():
        raise ValueError("phone boot changed")
    hashes = {line.split()[1]: line.split()[0] for line in (raw / "HASHES.txt").read_text().splitlines()}
    if hashes != read(root / "EXPECTED_PHONE_HASHES.json"):
        raise ValueError("phone identity changed")
    if hashes[ORIGINAL + "/llama-ffn-split-worker"] != ORIGINAL_WORKER_SHA or hashes[ORIGINAL + "/libggml-vulkan.so"] != ORIGINAL_VULKAN_SHA:
        raise ValueError("original runtime changed")
    save(root / "IDENTITY.json", {"phone_hashes": hashes, "geometry": identity["geometry"],
         "boot_id": (raw / "BOOT_AFTER.txt").read_text().strip(), "cpu_reference": read(root / "REFERENCE_PROVENANCE.json")})
    first_outputs = None
    summaries = {}
    fusion_coverage = {}
    cpu_affinity = {}
    dual_coverage = {}
    for arm in config["arms"]:
        source = raw / arm["name"]
        for name in ("NETCAT_EXIT.txt", "WORKER_EXIT.txt"):
            if (source / name).read_text().strip() != "0":
                raise ValueError("worker/capture did not exit normally")
        log = (source / "worker.log").read_text()
        expected_backend = "CPU" if arm.get("backend") == "CPU" else "PowerVR D-Series DXT-48-1536 MC1"
        if f"[ffn-worker] ready backend={expected_backend} layers=6 " not in log:
            raise ValueError("worker backend differs")
        if "cpu_threads" in arm:
            marker = (f"S42PIXELCPU threads={arm['cpu_threads']} mask={arm.get('cpu_mask', '0')} "
                      f"persistent={int(arm.get('cpu_pool', False))} poll=0")
            if marker not in log:
                raise ValueError("CPU configuration differs")
            mask = int(arm.get("cpu_mask", "0"), 16)
            if mask:
                status = (source / "THREAD_STATUS.txt").read_text()
                if arm.get("worker") == "cpu_gpu":
                    allowed = re.findall(r"Name:\s*llama-ffn-split\nPid:\s*\d+\nCpus_allowed_list:\s*(\S+)", status)
                else:
                    allowed = re.findall(r"^Cpus_allowed_list:\s*(\S+)", status, re.MULTILINE)
                expected = [str(i) for i in range(8) if mask & (1 << i)]
                cpu_affinity[arm["name"]] = {"status": "PASS" if sorted(allowed) == expected else "FAIL",
                                             "expected": expected, "actual": allowed}
        if "gpu_host_mask" in arm:
            allowed = re.findall(r"Name:\s*pixel-gpu-host\nPid:\s*\d+\nCpus_allowed_list:\s*(\S+)",
                                 (source / "THREAD_STATUS.txt").read_text())
            if len(allowed) != 1:
                raise ValueError("GPU host thread missing")
            actual = set()
            for item in allowed[0].split(","):
                bounds = [int(n) for n in item.split("-")]
                actual.update(range(bounds[0], bounds[-1] + 1))
            expected = {i for i in range(8) if int(arm["gpu_host_mask"], 16) & (1 << i)}
            if actual != expected:
                raise ValueError("GPU host affinity differs")
            cpu_affinity[arm["name"] + "-gpu-host"] = {"status": "PASS", "actual": sorted(actual)}
        if "Timings:" in log or "PIXEL_FFN_STAGE" in log:
            raise ValueError("profiling enabled")
        if "wg" in arm and f"S42PIXELGEMV wg={arm['wg']} rows={arm['rows']} subgroup={arm['subgroup']} cols=1" not in log:
            raise ValueError("kernel specialization not selected")
        if "shader" in arm and f"S42PIXELSHADER name={arm['shader']} precision=f32 cols=1" not in log:
            raise ValueError("shader not selected")
        if "cpu_half_columns" in arm:
            marker = f"S42PIXELRATIO cpu_per_half={arm['cpu_half_columns']} gpu_per_half={8704-arm['cpu_half_columns']} grain=64"
            if marker not in log:
                raise ValueError("fine ratio configuration differs")
        fusion_counts = re.findall(r"^S42PIXELFUSION op=MUL_MAT_SWIGLU dispatches=(\d+)$", log, re.MULTILINE)
        if arm.get("fuse_swiglu"):
            expected_dispatches = len(identity["geometry"]["layers"]) * 4 + sum(case["columns"] // 4352 for case in cases)
            if len(fusion_counts) != 1 or not 0 < int(fusion_counts[0]) <= expected_dispatches:
                raise ValueError(f"invalid SwiGLU dispatch count: {fusion_counts}, maximum {expected_dispatches}")
            fusion_coverage[arm["name"]] = {"status": "PASS" if int(fusion_counts[0]) == expected_dispatches else "FAIL",
                                           "expected_dispatches": expected_dispatches, "actual_dispatches": int(fusion_counts[0])}
        elif fusion_counts:
            raise ValueError("unexpected SwiGLU fusion")
        data = (source / "RESPONSES.bin").read_bytes()
        hello = HELLO_RESPONSE.unpack(data[:HELLO_RESPONSE.size])
        artifact = bytes.fromhex(identity["geometry"]["artifact_sha256"].removeprefix("sha256:"))
        expected = (MAGIC, 6, 2, 0, 3, 5120, 17408, 0, 17408, 1, 6, 16515072)
        if hello[:12] != expected or hello[13:] != (4352, 4, 0, artifact):
            raise ValueError("HELLO mismatch")
        directory = root / arm["name"]
        directory.mkdir()
        offset, rows, outputs = HELLO_RESPONSE.size, [], []
        for case in cases:
            response = EXEC_RESPONSE.unpack(data[offset:offset + EXEC_RESPONSE.size])
            offset += EXEC_RESPONSE.size
            output = data[offset:offset + 10240]
            offset += 10240
            if response[:9] != (MAGIC, 6, 4, 0, 0, case["id"], case["layer"], 5120, 10240) or response[10:12] != (case["columns"], 1):
                raise ValueError("response shape/order mismatch")
            if len(output) != 10240 or fnv(output) != response[9] or response[12] <= 0:
                raise ValueError("response payload/timing mismatch")
            values = np.frombuffer(output, dtype="<f2").astype(np.float64)
            cpu = np.fromfile(root / "cpu" / f"output-{case['id']:03d}.f16", dtype="<f2").astype(np.float64)
            relative = float(np.linalg.norm(values - cpu) / max(np.linalg.norm(cpu), 1e-30))
            if not np.isfinite(values).all() or relative > 0.01:
                raise ValueError("numerical mismatch")
            (directory / f"output-{case['id']:03d}.f16").write_bytes(output)
            rows.append({**case, "warm": case["repeat"] >= config.get("warmup_repeats", 2), "worker_us": response[12],
                         "output_sha256": sha(output), "relative_l2": relative})
            outputs.append(output)
        if offset != len(data):
            raise ValueError("extra response bytes")
        first_outputs = outputs if first_outputs is None else first_outputs
        exact = sum(a == b for a, b in zip(outputs, first_outputs))
        if arm.get("require_exact") and exact != len(cases):
            raise ValueError("control bytes differ")
        (directory / "CALLS.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary = {**summarize(rows), "arm": arm, "calls": len(cases), "exact_calls_vs_control": exact}
        if "gpu_block_mask" in arm:
            dual_coverage[arm["name"]] = audit_dual(log, arm, rows)
        if fusion_counts:
            summary["swiglu_fused_dispatches"] = int(fusion_counts[0])
        save(directory / "SUMMARY.json", summary)
        summaries[arm["name"]] = summary
    save(root / "CLEANUP.json", {"status": "PASS", "boot_id": (raw / "BOOT_AFTER.txt").read_text().strip(),
         "processes": (raw / "PROCESSES_AFTER.txt").read_text(), "adb_forward_created": False})
    save(root / "RESULT.json", {"status": "PASS", "phone_calls": len(cases) * len(config["arms"]),
         "calls_per_arm": len(cases), "arms": summaries, "measurement_scope": config["measurement_scope"]})
    if fusion_coverage:
        save(root / "FUSION_COVERAGE.json", {"status": "PASS" if all(c["status"] == "PASS" for c in fusion_coverage.values()) else "FAIL",
             "arms": fusion_coverage})
    if cpu_affinity:
        save(root / "CPU_AFFINITY.json", {"status": "PASS" if all(row["status"] == "PASS" for row in cpu_affinity.values()) else "FAIL",
                                         "arms": cpu_affinity})
    if dual_coverage:
        save(root / "DUAL_COVERAGE.json", {
            "status": "PASS" if all(row["status"] == "PASS" for row in dual_coverage.values()) else "FAIL",
            "arms": dual_coverage})
    result = analyze(root)
    result["measurement_scope"] = config["measurement_scope"]
    save(root / "SWEEP_AUDIT.json", result)
    print(json.dumps({"status": result["status"], "phone_calls": result["phone_calls"],
                      "maximum_relative_l2": result["maximum_relative_l2"], "fusion_coverage": fusion_coverage,
                      "comparisons": result["comparisons"]}, indent=2))


def audit_dual(log, arm, rows):
    markers = [dict((key, int(value)) for key, value in re.findall(r"(\w+)=(\d+)", line))
               for line in log.splitlines() if line.startswith("S43DUALFFN ")]
    if len(markers) != len(rows):
        raise ValueError("missing dual execution records")
    nonoverlapping = []
    for marker, row in zip(markers, rows):
        mask = int(arm["gpu_block_mask"], 16)
        if "cpu_half_columns" in arm:
            gpu_columns = (8704 - arm["cpu_half_columns"]) * (row["columns"] // 8704)
        else:
            gpu_columns = sum(4352 for i in range(4 - row["columns"] // 4352, 4) if mask & (1 << i))
        if (marker["request"], marker["layer"], marker["tokens"], marker["columns"],
                marker["primary_columns"], marker["secondary_columns"]) != (
                row["id"], row["layer"], 1, row["columns"], row["columns"] - gpu_columns, gpu_columns):
            raise ValueError("CPU/GPU weight partition differs")
        overlap = max(0, min(marker["primary_us"], marker["secondary_end_us"]) -
                      max(marker["primary_start_us"], marker["secondary_start_us"]))
        if (marker["overlap_us"] != overlap or marker["secondary_end_us"] - marker["secondary_start_us"] != marker["secondary_us"] or
                marker["total_us"] != marker["primary_us"] + marker["wait_us"] + marker["merge_us"] or
                marker["total_us"] > row["worker_us"]):
            raise ValueError("dual timing proof failed")
        if 0 < gpu_columns < row["columns"] and overlap <= 0:
            nonoverlapping.append(row["id"])
    summaries = {}
    for columns in (8704, 17408):
        warm = [marker for marker, row in zip(markers, rows) if row["warm"] and row["columns"] == columns]
        summaries[str(columns)] = {
            "calls": len(warm), "cpu_columns": warm[0]["primary_columns"],
            "gpu_columns": warm[0]["secondary_columns"],
            "mean_ms": {key.removesuffix("_us"): statistics.mean(m[key] for m in warm) / 1000
                        for key in ("primary_us", "secondary_us", "wait_us", "merge_us", "total_us", "overlap_us")},
            "minimum_overlap_ms": min(m["overlap_us"] for m in warm) / 1000,
            "overlapping_warm_calls": sum(m["overlap_us"] > 0 for m in warm),
        }
    return {"status": "FAIL" if nonoverlapping else "PASS", "calls": len(markers), "widths": summaries,
            "partition_and_timing_status": "PASS", "overlapping_calls": len(markers) - len(nonoverlapping),
            "nonoverlapping_request_ids": nonoverlapping,
            "scope": "overlap of backend execution branches; not hardware kernel timestamps"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "audit"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--software", type=Path)
    parser.add_argument("--worker-software", type=Path)
    parser.add_argument("--previous-software", type=Path)
    parser.add_argument("--cpu-software", type=Path)
    parser.add_argument("--dual-software", type=Path)
    parser.add_argument("--phone-dir")
    arguments = parser.parse_args()
    (prepare if arguments.action == "prepare" else audit)(arguments)
