"""Extend the finite phone-local sweep with independent rows and packed artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
from types import SimpleNamespace

import numpy as np

import pixel_local_sweep as local
from qualify_op11_tcp import EXEC_REQUEST, EXEC_RESPONSE, HELLO_REQUEST, HELLO_RESPONSE, MAGIC, fnv
from tune_pixel_gemv import ORIGINAL, PHONE_MODEL


BASE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(args):
    root = args.root.resolve()
    options = SimpleNamespace(root=root, reference=BASE / "physical/pixel10pro-gemv-confirm-1/run1",
        config=args.config, software=args.software, previous_software=BASE / "software/pixel10pro-dense-gemv-ratio-v1",
        worker_software=BASE / "software/pixel10pro-cpu-tune-v3", dual_software=args.dual_software,
        cpu_software=BASE / "software/pixel10pro-cpu-tune-v3", phone_dir=args.phone_dir)
    local.prepare(options)
    config = read(root / "CONFIG.json")
    staging = {name: str(root / name) for name in ("EXPECTED_HASHES.sha256", "REQUESTS.bin", "RUN_PHONE.sh")}
    for dest, source in {
        "libggml-vulkan.so": args.software / "libggml-vulkan.so",
        "previous/libggml-vulkan.so": options.previous_software / "libggml-vulkan.so",
        "llama-ffn-split-worker-dual": args.dual_software / "llama-ffn-split-worker",
        "llama-ffn-split-worker": options.worker_software / "llama-ffn-split-worker",
        "cpu-affinity/libggml-cpu.so": options.cpu_software / "libggml-cpu.so",
    }.items():
        staging[dest] = str(source.resolve())
    script = (root / "RUN_PHONE.sh").read_text()
    script = script.replace('< REQUESTS.bin', '< "$request_packet"')
    original_count = 12 * config["repeats"]
    geometry = read(root / "REFERENCE_IDENTITY.json")["geometry"]
    artifact = geometry["artifact_sha256"]
    cases = read(root / "CASES.json")
    for case in cases:
        case.update(tokens=1, row_indices=[0])
    batch = bool(config.get("batches"))
    inputs = {}
    if batch:
        if config["arms"][0]["name"] != "reference-cpu" or config["arms"][0].get("backend") != "CPU":
            raise ValueError("batch validation needs independent CPU rows first")
        for layer in geometry["layers"]:
            original = np.fromfile(root / f"input-layer{layer}.f16", dtype="<f2")
            for index in range(8):
                payload = (np.roll(original, 127 * index).astype(np.float32) * (1 - 0.0625 * index)).astype("<f2").tobytes()
                inputs[layer, index] = payload
                (root / f"input-layer{layer}-row{index}.f16").write_bytes(payload)
            if len({sha(inputs[layer, j]) for j in range(8)}) != 8:
                raise ValueError("input rows must be distinct")
        cases = []
        for repeat in range(config["repeats"]):
            for tokens in config["batches"]:
                if tokens not in (1, 2, 4, 8):
                    raise ValueError("unsupported batch")
                for layer in geometry["layers"]:
                    for columns in (8704, 17408):
                        cases.append(dict(id=len(cases)+1, layer=layer, columns=columns,
                                          tokens=tokens, row_indices=list(range(tokens)), repeat=repeat))
    reference_cases = [dict(id=i+1, layer=layer, columns=columns, tokens=1, row_indices=[j], repeat=0)
        for i, (layer, columns, j) in enumerate((layer, columns, j)
            for layer in geometry["layers"] for columns in (8704, 17408) for j in range(8))] if batch else cases
    save(root / "CASES.json", cases)
    save(root / "INDEPENDENT_REFERENCE_CASES.json", reference_cases)
    max_tokens = 8 if batch else 4
    packet_files = {}
    for arm in config["arms"]:
        arm_cases = reference_cases if arm["name"] == "reference-cpu" else cases
        arm_artifact = arm.get("artifact_sha256", artifact)
        if batch or arm.get("packed_weights"):
            packet = bytearray(HELLO_REQUEST.pack(MAGIC, 6, 1, sum(1 << n for n in geometry["layers"]),
                5120, 17408, 3, max_tokens, bytes.fromhex(arm_artifact.removeprefix("sha256:"))))
            for case in arm_cases:
                payload = b"".join(inputs[case["layer"], j] for j in case["row_indices"]) if batch else (
                    root / f"input-layer{case['layer']}.f16").read_bytes()
                packet.extend(EXEC_REQUEST.pack(MAGIC, 6, 3, case["id"], case["layer"], 5120,
                    len(payload), fnv(payload), case["columns"], case["tokens"]) + payload)
            packet_name = f"REQUESTS-{arm['name']}.bin"
            (root / packet_name).write_bytes(packet)
            staging[packet_name] = str(root / packet_name)
        else:
            packet_name = "REQUESTS.bin"
        packet_files[arm["name"]] = {"file": packet_name, "sha256": sha((root / packet_name).read_bytes()), "calls": len(arm_cases)}
        lines = script.splitlines(keepends=True)
        found = False
        for index, line in enumerate(lines):
            if line.startswith("run_arm " + arm["name"] + " "):
                found = True
                line = line.replace(f"--max-requests {original_count}", f"--max-requests {len(arm_cases)}")
                line = line.replace("--max-tokens 4", f"--max-tokens {max_tokens}")
                if arm.get("packed_weights"):
                    line = line.replace(PHONE_MODEL, arm["model_path"]).replace(artifact, arm_artifact)
                lines[index] = f"request_packet={shlex.quote(packet_name)}\n" + line
        if not found:
            raise ValueError("arm launch anchor differs")
        script = "".join(lines)
    expected = read(root / "EXPECTED_PHONE_HASHES.json")
    for arm in config["arms"]:
        if arm.get("packed_weights"):
            expected[arm["model_path"]] = arm["model_sha256"]
    save(root / "EXPECTED_PHONE_HASHES.json", expected)
    (root / "EXPECTED_HASHES.sha256").write_text("".join(f"{h}  {p}\n" for p, h in expected.items()))
    script = re.sub(r"^sha256sum .* > raw/HASHES.txt", "sha256sum " + shlex.join(expected) + " > raw/HASHES.txt", script, flags=re.MULTILINE)
    (root / "RUN_PHONE.sh").write_text(script)
    save(root / "PACKETS.json", packet_files)
    save(root / "STAGING.json", staging)
    (root / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(f"SUITE PREPARED {sum(p['calls'] for p in packet_files.values())} calls")


def affinity(source, arm):
    status = (source / "THREAD_STATUS.txt").read_text()
    mask = int(arm.get("cpu_mask", "0"), 16)
    if mask and arm.get("cpu_affinity_fix"):
        actual = re.findall(r"Name:\s*llama-ffn-split\nPid:\s*\d+\nCpus_allowed_list:\s*(\S+)", status)
        if sorted(actual) != [str(i) for i in range(8) if mask & (1 << i)]:
            raise ValueError("CPU affinity differs")
    if "gpu_host_mask" in arm:
        rows = re.findall(r"Name:\s*pixel-gpu-host\nPid:\s*\d+\nCpus_allowed_list:\s*(\S+)", status)
        if len(rows) != 1:
            raise ValueError("missing GPU helper")
        actual = set()
        for item in rows[0].split(","):
            pair = [int(n) for n in item.split("-")]
            actual.update(range(pair[0], pair[-1] + 1))
        if actual != {i for i in range(8) if int(arm["gpu_host_mask"], 16) & (1 << i)}:
            raise ValueError("GPU host affinity differs")


def audit(args):
    root = args.root
    config = read(root / "CONFIG.json")
    raw = root / "raw"
    if (raw / "DONE.txt").read_text().strip() != "PASS" or (root / "EXIT.txt").read_text().strip() != "0":
        raise ValueError("incomplete suite")
    if (raw / "BOOT_BEFORE.txt").read_bytes() != (raw / "BOOT_AFTER.txt").read_bytes():
        raise ValueError("boot changed")
    hashes = {line.split()[1]: line.split()[0] for line in (raw / "HASHES.txt").read_text().splitlines()}
    if hashes != read(root / "EXPECTED_PHONE_HASHES.json"):
        raise ValueError("identity changed")
    cases = read(root / "CASES.json")
    reference_cases = read(root / "INDEPENDENT_REFERENCE_CASES.json")
    reference = {}
    summaries = {}
    all_rows = {}
    batch = bool(config.get("batches"))
    geometry = read(root / "REFERENCE_IDENTITY.json")["geometry"]
    controls = {}
    for arm in config["arms"]:
        name = arm["name"]
        source = raw / name
        if any((source / filename).read_text().strip() != "0" for filename in ("NETCAT_EXIT.txt", "WORKER_EXIT.txt")):
            raise ValueError("abnormal worker/capture exit")
        affinity(source, arm)
        log = (source / "worker.log").read_text()
        backend = "CPU" if arm.get("backend") == "CPU" else "PowerVR D-Series DXT-48-1536 MC1"
        if f"[ffn-worker] ready backend={backend} layers=6 " not in log:
            raise ValueError("backend differs")
        if arm.get("coalesce_full") and "S42PIXELCOALESCE full_gemv=6 half_gemv=6 resident_factor=1.5" not in log:
            raise ValueError("coalescing not selected")
        if arm.get("packed_weights") and len(re.findall(r"^S42PIXELPACKED layer=", log, re.MULTILINE)) != 6:
            raise ValueError("packed path not selected")
        data = (source / "RESPONSES.bin").read_bytes()
        hello = HELLO_RESPONSE.unpack(data[:HELLO_RESPONSE.size])
        artifact = bytes.fromhex(arm.get("artifact_sha256", geometry["artifact_sha256"]).removeprefix("sha256:"))
        weight_type = 12 if arm.get("packed_weights") else 1
        if hello[:12] != (MAGIC, 6, 2, 0, 3, 5120, 17408, 0, 17408, weight_type, 6, 16515072) or hello[13:] != (4352, 8 if batch else 4, 0, artifact):
            raise ValueError(f"HELLO mismatch: {hello}")
        arm_cases = reference_cases if name == "reference-cpu" else cases
        offset = HELLO_RESPONSE.size
        rows = []
        for case in arm_cases:
            response = EXEC_RESPONSE.unpack(data[offset:offset+EXEC_RESPONSE.size])
            offset += EXEC_RESPONSE.size
            size = 10240 * case["tokens"]
            output = data[offset:offset+size]
            offset += size
            if response[:9] != (MAGIC, 6, 4, 0, 0, case["id"], case["layer"], 5120, size) or response[10:12] != (case["columns"], case["tokens"]):
                raise ValueError("response shape/order differs")
            if len(output) != size or fnv(output) != response[9] or response[12] <= 0:
                raise ValueError("invalid response/hash/time")
            values = np.frombuffer(output, dtype="<f2").astype(np.float64).reshape(case["tokens"], 5120)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite output")
            if name == "reference-cpu":
                reference[case["layer"], case["columns"], case["row_indices"][0]] = output
                relative = 0.0
            else:
                expected = b"".join(reference[case["layer"], case["columns"], j] for j in case["row_indices"]) if batch else (
                    root / "cpu" / f"output-{case['id']:03d}.f16").read_bytes()
                target = np.frombuffer(expected, dtype="<f2").astype(np.float64).reshape(case["tokens"], 5120)
                relative = float(np.max(np.linalg.norm(values-target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-30)))
            key = (case["layer"], case["columns"], tuple(case["row_indices"]))
            if arm.get("require_exact") and key in controls and controls[key] != output:
                raise ValueError("control bytes changed")
            if name == config["comparison_controls"][0]:
                controls[key] = output
            rows.append({**case, "worker_us": response[12], "warm": name != "reference-cpu" and case["repeat"] >= config.get("warmup_repeats", 2),
                         "relative_l2": relative, "output_sha256": sha(output)})
        if offset != len(data):
            raise ValueError("extra bytes")
        dual = [dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", line)) for line in log.splitlines() if line.startswith("S43DUALFFN ")]
        if arm.get("gpu_block_mask"):
            if len(dual) != len(rows):
                raise ValueError("missing dual records")
            for marker, row in zip(dual, rows):
                gpu = (8704-arm["cpu_half_columns"]) * row["columns"] // 8704
                if tuple(marker[k] for k in ("request", "layer", "tokens", "columns", "primary_columns", "secondary_columns")) != (
                        row["id"], row["layer"], row["tokens"], row["columns"], row["columns"]-gpu, gpu):
                    raise ValueError("dual partition differs")
                overlap = max(0, min(marker["primary_us"], marker["secondary_end_us"]) - max(marker["primary_start_us"], marker["secondary_start_us"]))
                if overlap != marker["overlap_us"] or marker["total_us"] > row["worker_us"]:
                    raise ValueError("invalid dual timing")
        groups = {}
        for tokens in sorted({row["tokens"] for row in rows}):
            for width in (8704, 17408):
                selected = [row for row in rows if row["tokens"] == tokens and row["columns"] == width and row["warm"]]
                if not selected:
                    continue
                times = np.array([row["worker_us"] for row in selected]) / 1000
                groups[f"b{tokens}-c{width}"] = dict(calls=len(selected), mean_ms=float(times.mean()), median_ms=float(np.median(times)),
                    p99_ms=float(np.percentile(times, 99)), per_row_ms=float(times.mean()/tokens))
        worst = max(row["relative_l2"] for row in rows)
        summaries[name] = dict(numerical_status="PASS" if worst <= 0.01 else "FAIL", max_relative_l2=worst,
            calls=len(rows), groups=groups, dual_calls=len(dual), overlapping_calls=sum(m["overlap_us"] > 0 for m in dual),
            arm=arm)
        all_rows[name] = rows
    save(root / "SUITE_CALLS.json", all_rows)
    save(root / "SUITE_RESULT.json", dict(status="PASS" if all(r["numerical_status"] == "PASS" for r in summaries.values()) else "FAIL",
        phone_calls=sum(len(rows) for rows in all_rows.values()), arms=summaries, boot_id=(raw / "BOOT_AFTER.txt").read_text().strip(),
        scope="phone-local worker; synthetic independent input rows for batch tests; no transport/energy/full-model result"))
    print(json.dumps({name: {k: v for k, v in row.items() if k != "arm"} for name, row in summaries.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "audit"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--software", type=Path)
    parser.add_argument("--dual-software", type=Path)
    parser.add_argument("--phone-dir")
    arguments = parser.parse_args()
    (prepare if arguments.action == "prepare" else audit)(arguments)
