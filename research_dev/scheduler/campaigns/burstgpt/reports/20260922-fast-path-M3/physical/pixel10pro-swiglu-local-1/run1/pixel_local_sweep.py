"""Prepare and audit a phone-local finite sweep using archived CPU references."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
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
    count = 12 * config["repeats"]
    identity = read(reference / "IDENTITY.json")
    geometry = SimpleNamespace(**identity["geometry"])
    cases = read(reference / "CASES.json")[:count]
    cpu_rows = [json.loads(line) for line in (reference / "cpu/CALLS.jsonl").read_text().splitlines()][:count]
    if read(reference / "RESULT.json")["status"] != "PASS" or len(cases) != count or len(cpu_rows) != count:
        raise ValueError("insufficient qualified reference cases")
    if not args.phone_dir.startswith("/data/local/tmp/s42-pixel10pro-gemv-tune-"):
        raise ValueError("private phone directory required")
    if any(arm["quantum"] != 4352 or arm.get("worker", "original") != "original" for arm in config["arms"]):
        raise ValueError("this local sweep requires the qualified worker and quantum")
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
run_arm() {
    name="$1"
    shift
    mkdir "raw/$name" || return 7
    dumpsys battery > "raw/$name/BATTERY_BEFORE.txt"
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
    nc -n -w 5 -W 120 127.0.0.1 27141 < REQUESTS.bin > "raw/$name/RESPONSES.bin"
    network_status=$?
    echo "$network_status" > "raw/$name/NETCAT_EXIT.txt"
    if [ "$network_status" != 0 ]; then echo "FAIL $name capture"; return 10; fi
    wait "$worker_pid"
    worker_status=$?
    echo "$worker_status" > "raw/$name/WORKER_EXIT.txt"
    if [ "$worker_status" != 0 ]; then echo "FAIL $name worker"; return 11; fi
    dumpsys battery > "raw/$name/BATTERY_AFTER.txt"
    echo "PASS $name"
}
"""
    script = header.replace("PHONE_DIR", shlex.quote(args.phone_dir)).replace("HASH_FILES", shlex.join(expected))
    for arm in config["arms"]:
        runtime = ORIGINAL if arm["runtime"] == "original" else args.phone_dir + ":" + ORIGINAL
        command = ["env", "-u", "GGML_VK_PERF_LOGGER", "-u", "GGML_VK_DISABLE_FUSION", "-u", "S42_PIXEL_PROFILE_OPS",
                   "-u", "S42_PIXEL_F16_WG", "-u", "S42_PIXEL_F16_ROWS", "-u", "S42_PIXEL_F16_SUBGROUP",
                   "-u", "S42_PIXEL_F16_SHADER", "-u", "S42_PIXEL_FUSE_SWIGLU", "LD_LIBRARY_PATH=" + runtime]
        for key, variable in (("wg", "WG"), ("rows", "ROWS"), ("subgroup", "SUBGROUP"), ("shader", "SHADER")):
            if key in arm:
                command.append(f"S42_PIXEL_F16_{variable}={arm[key]}")
        if arm.get("fuse_swiglu"):
            if arm["runtime"] != "tuned" or "shader" not in arm:
                raise ValueError("SwiGLU fusion requires a custom shader")
            command.append("S42_PIXEL_FUSE_SWIGLU=1")
        command += worker_command(geometry, ORIGINAL + "/llama-ffn-split-worker", PHONE_MODEL, "Vulkan0", 27141, count)
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
    for arm in config["arms"]:
        source = raw / arm["name"]
        for name in ("NETCAT_EXIT.txt", "WORKER_EXIT.txt"):
            if (source / name).read_text().strip() != "0":
                raise ValueError("worker/capture did not exit normally")
        log = (source / "worker.log").read_text()
        if "Timings:" in log or "PIXEL_FFN_STAGE" in log:
            raise ValueError("profiling enabled")
        if "wg" in arm and f"S42PIXELGEMV wg={arm['wg']} rows={arm['rows']} subgroup={arm['subgroup']} cols=1" not in log:
            raise ValueError("kernel specialization not selected")
        if "shader" in arm and f"S42PIXELSHADER name={arm['shader']} precision=f32 cols=1" not in log:
            raise ValueError("shader not selected")
        fusion_counts = re.findall(r"^S42PIXELFUSION op=MUL_MAT_SWIGLU dispatches=(\d+)$", log, re.MULTILINE)
        if arm.get("fuse_swiglu"):
            expected_dispatches = len(identity["geometry"]["layers"]) * 4 + sum(case["columns"] // 4352 for case in cases)
            if fusion_counts != [str(expected_dispatches)]:
                raise ValueError(f"SwiGLU dispatch count differs: {fusion_counts}, expected {expected_dispatches}")
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
            rows.append({**case, "warm": case["repeat"] >= 2, "worker_us": response[12],
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
        if fusion_counts:
            summary["swiglu_fused_dispatches"] = int(fusion_counts[0])
        save(directory / "SUMMARY.json", summary)
        summaries[arm["name"]] = summary
    save(root / "CLEANUP.json", {"status": "PASS", "boot_id": (raw / "BOOT_AFTER.txt").read_text().strip(),
         "processes": (raw / "PROCESSES_AFTER.txt").read_text(), "adb_forward_created": False})
    save(root / "RESULT.json", {"status": "PASS", "phone_calls": len(cases) * len(config["arms"]),
         "calls_per_arm": len(cases), "arms": summaries, "measurement_scope": config["measurement_scope"]})
    result = analyze(root)
    result["measurement_scope"] = config["measurement_scope"]
    save(root / "SWEEP_AUDIT.json", result)
    print(json.dumps({"status": result["status"], "phone_calls": result["phone_calls"],
                      "maximum_relative_l2": result["maximum_relative_l2"], "comparisons": result["comparisons"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "audit"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--software", type=Path)
    parser.add_argument("--phone-dir")
    arguments = parser.parse_args()
    (prepare if arguments.action == "prepare" else audit)(arguments)
