#!/usr/bin/env python3
"""S14 Checkpoint B: phone BGE encode atlas on OP15/HTP0 (v81) and OP12/HTP0 (v75).

Runs the same bge-small-en-v1.5 encode shapes as CP-A on each phone's HTP backend,
using the freshly built llama-embedding (BGEPROF timing + PLACEMENTCERT). Emits, per
shape and per device, a same-run latency profile AND a scheduled-placement certificate.

Fail-closed unless: PLACEMENTCERT status == SCHEDULED_PLACEMENT_OK, missing_buffer == 0,
HTP0 compute nodes > 0, every non-HTP compute op is the declared GET_ROWS CPU exception,
BGEPROF finite, and output cosine vs the CPU f32 reference >= gate. Records binary sha256,
device/build identity, and thermal state. Writes an APPEND-ONLY atlas extension; the frozen
CP0 island_catalog.json is not mutated.

Phone energy remains UNKNOWN; this is latency + placement only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from bge_corpus import make_prompt, write_corpus

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
GGUF = str(REPO / "models/bge-small-en-v1.5-f16.gguf")
CPU_BIN = str(REPO / "build-cpu/bin/llama-embedding")
CPU_LIB = str(REPO / "build-cpu/bin")
SERVER_PROFILE = HERE / "bge_server_result.json"
MODEL_SHA = "4cd429b83d2805e4028d96f6174b153bc87657808ba9f3b3c4f84d374b481e03"

PHONES = {
    "op15": {"serial": "3C15AU002CL00000", "hexagon": "v81", "dir": "/data/local/tmp/bge",
             "skel": "libggml-htp-v81.so"},
    "op12": {"serial": "5ae7a43d", "hexagon": "v75", "dir": "/data/local/tmp/ls-s14",
             "skel": "libggml-htp-v75.so"},
}
REMOTE_MODEL = "/data/local/tmp/ls-npu/bge-small-en-v1.5-f16.gguf"
DECLARED_CPU_OPS = {"GET_ROWS"}  # the one declared f16 embedding-table exception


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_server_exact(path: Path) -> dict[int, int]:
    data = json.loads(path.read_text())
    if data.get("schema") not in {"s14-bge-server-control-v2", "s14-bge-server-control-v3"}:
        raise ValueError("unsupported server profile schema")
    if data.get("model_sha256") != MODEL_SHA:
        raise ValueError("server profile model digest mismatch")
    rows = data.get("correctness_cuda_vs_cpu")
    if not isinstance(rows, dict) or not rows:
        raise ValueError("server profile has no correctness rows")
    result = {}
    for target, row in rows.items():
        exact = row.get("seq_len_exact") if isinstance(row, dict) else None
        if isinstance(exact, bool) or not isinstance(exact, int) or exact <= 0:
            raise ValueError(f"invalid server exact length for target {target}")
        result[int(target)] = exact
    return result


def adb_sh(serial: str, cmd: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", "-s", serial, "shell", cmd], capture_output=True, text=True, timeout=timeout)


def thermal_snapshot(serial: str) -> dict:
    command = (
        "for z in /sys/class/thermal/thermal_zone*; do "
        "ty=$(cat $z/type 2>/dev/null); t=$(cat $z/temp 2>/dev/null); "
        "case $ty in nsphmx-*) echo $ty=$t;; esac; done"
    )
    process = adb_sh(serial, command)
    sensors = {}
    for line in process.stdout.splitlines():
        if "=" not in line:
            continue
        name, raw = line.split("=", 1)
        try:
            value = int(raw)
        except ValueError:
            continue
        if 10_000 <= value <= 120_000:
            sensors[name] = value
    return {
        "sensor_class": "nsphmx-*",
        "sensors_millic": dict(sorted(sensors.items())),
        "max_millic": max(sensors.values()) if sensors else None,
        "valid": process.returncode == 0 and bool(sensors),
    }


def remote_sha(serial: str, path: str) -> str:
    return (adb_sh(serial, f"sha256sum {path} 2>/dev/null").stdout.split() or ["?"])[0]


def cpu_ref(words: int) -> list[float]:
    corpus = HERE / "logs_bge" / f"cpu_ref_{words}.txt"
    corpus.parent.mkdir(exist_ok=True)
    write_corpus(corpus, make_prompt(words), 1)
    env = {"LD_LIBRARY_PATH": CPU_LIB, "PATH": "/usr/bin:/bin", "BGEPROF_REPS": "1", "BGEPROF_WARMUP": "0"}
    cmd = [CPU_BIN, "-m", GGUF, "--pooling", "cls", "--embd-normalize", "2", "-fa", "off",
           "-f", str(corpus), "--parallel", "2", "-c", "1024", "-b", "1024", "--embd-output-format", "array"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("BGEPROF ")), None)
    return json.loads(line[len("BGEPROF "):])["emb0"] if line else []


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def pctl(xs: list[float], q: float) -> float:
    s = sorted(xs); k = (len(s) - 1) * q; lo = int(k)
    return s[lo] if lo + 1 >= len(s) else s[lo] + (k - lo) * (s[lo + 1] - s[lo])


def run_shape(serial: str, wdir: str, words: int, batch: int, ctx: int, mbuf: int,
              reps: int, warmup: int, remote_bin: str) -> tuple[dict | None, dict | None, str]:
    remote_corpus = "/data/local/tmp/bge_cp_b.txt"
    prompt = make_prompt(words)
    lines = "\\n".join([prompt] * batch)
    adb_sh(serial, f"printf '{lines}\\n' > {remote_corpus}")
    env = (f"BGEPROF_REPS={reps} BGEPROF_WARMUP={warmup} LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
           f"GGML_HEXAGON_MBUF={mbuf}")
    cmd = (f"cd {wdir} && {env} ./{remote_bin} -m {REMOTE_MODEL} --device HTP0 -ngl 99 "
           f"--pooling cls --embd-normalize 2 -fa off -f {remote_corpus} --parallel {max(batch,2)} "
           f"-c {ctx} -b {ctx} --embd-output-format array")
    p = adb_sh(serial, cmd)
    text = p.stdout + "\n" + p.stderr
    bge = cert = None
    for ln in text.splitlines():
        if ln.startswith("BGEPROF "):
            bge = json.loads(ln[len("BGEPROF "):])
        elif ln.startswith("PLACEMENTCERT "):
            cert = json.loads(ln[len("PLACEMENTCERT "):])
    return bge, cert, text[-400:]


def cert_ok(cert: dict) -> tuple[bool, str]:
    if cert is None:
        return False, "no PLACEMENTCERT"
    if cert.get("status") != "SCHEDULED_PLACEMENT_OK":
        return False, f"status {cert.get('status')}"
    if cert.get("missing_buffer_compute_nodes", 1) != 0:
        return False, "missing_buffer != 0"
    htp = sum(v for k, v in cert.get("by_buffer", {}).items() if "HTP" in k)
    if htp <= 0:
        return False, "HTP0 compute nodes == 0"
    for entry in cert.get("non_htp_ops", []):
        op = entry.split("@", 1)[0]
        if op not in DECLARED_CPU_OPS:
            return False, f"undeclared CPU op {op}"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="32:25,128:105,512:400")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mbuf", type=int, default=3072)
    ap.add_argument("--cosine-gate", type=float, default=0.99)
    ap.add_argument("--cov-gate", type=float, default=0.05)
    ap.add_argument("--thermal-max-millic", type=int, default=85_000)
    ap.add_argument("--server-profile", type=Path, default=SERVER_PROFILE)
    ap.add_argument("--phones", default="op15,op12")
    ap.add_argument("--remote-bin", default="llama-embedding.s14v2")
    ap.add_argument("--output", default=str(HERE / "cp_b_phone_bge_result.json"))
    args = ap.parse_args()

    seqs = [(int(a), int(b)) for a, b in (s.split(":") for s in args.seqs.split(","))]
    batches = [int(x) for x in args.batches.split(",")]
    server_exact = load_server_exact(args.server_profile)
    requested_targets = {target for target, _ in seqs}
    if not requested_targets.issubset(server_exact):
        raise ValueError("server profile does not cover every requested phone target")
    server_exact = {target: server_exact[target] for target in requested_targets}
    refs = {L: cpu_ref(w) for L, w in seqs}

    devices, shapes, failures = {}, [], []
    for pname in args.phones.split(","):
        ph = PHONES[pname]
        serial = ph["serial"]
        thermal_start = thermal_snapshot(serial)
        devices[pname] = {
            "serial": serial, "hexagon": ph["hexagon"],
            "binary_sha256": remote_sha(serial, f"{ph['dir']}/{args.remote_bin}"),
            "skel_sha256": remote_sha(serial, f"{ph['dir']}/{ph['skel']}"),
            "model_sha256": remote_sha(serial, REMOTE_MODEL),
            "thermal_npu_start": thermal_start,
        }
        if devices[pname]["model_sha256"] != MODEL_SHA:
            failures.append(f"{pname}: remote model digest mismatch")
        if not thermal_start["valid"] or thermal_start["max_millic"] > args.thermal_max_millic:
            failures.append(f"{pname}: invalid start thermal sample")
        for L, words in seqs:
            for B in batches:
                ctx = (((max(B, 2) * (L + 16)) + 63) // 32) * 32
                all_us, cert_last, exact, ok = [], None, None, 0
                emb0 = None
                for _ in range(args.procs):
                    bge, cert, tail = run_shape(serial, ph["dir"], words, B, ctx, args.mbuf,
                                                args.reps, args.warmup, args.remote_bin)
                    good, why = cert_ok(cert)
                    if bge is None:
                        failures.append(f"{pname} L{L}B{B}: no BGEPROF ({tail[-120:]})")
                        continue
                    if not good:
                        failures.append(f"{pname} L{L}B{B}: cert {why}")
                        continue
                    if not bge["finite"] or bge["batch"] != B:
                        failures.append(f"{pname} L{L}B{B}: finite={bge['finite']} batch={bge['batch']}")
                        continue
                    measured_exact = bge["total_tokens"] // B
                    if measured_exact != server_exact[L]:
                        failures.append(
                            f"{pname} L{L}B{B}: exact tokens {measured_exact} != server {server_exact[L]}"
                        )
                        continue
                    all_us.extend(bge["us"]); cert_last = cert; exact = bge["total_tokens"] // B
                    emb0 = bge["emb0"]; ok += 1
                if not all_us:
                    shapes.append({"device": pname, "seq_len": L, "batch": B, "status": "NO_SAMPLES"})
                    continue
                cos = cosine(emb0, refs[L]) if emb0 else 0.0
                if cos < args.cosine_gate:
                    failures.append(f"{pname} L{L}B{B}: cosine {cos:.4f} < {args.cosine_gate}")
                p50 = pctl(all_us, 0.50)
                mean = sum(all_us) / len(all_us)
                cov = (sum((x - mean) ** 2 for x in all_us) / len(all_us)) ** 0.5 / mean
                if cov > args.cov_gate:
                    failures.append(f"{pname} L{L}B{B}: CoV {cov:.4f} > {args.cov_gate}")
                shapes.append({
                    "device": pname, "hexagon": ph["hexagon"], "seq_len_target": L,
                    "seq_len_exact": exact, "batch": B, "ok_procs": ok, "n_samples": len(all_us),
                    "lat_us_p50": round(p50, 1), "lat_us_p95": round(pctl(all_us, 0.95), 1),
                    "lat_us_p99": round(pctl(all_us, 0.99), 1),
                    "cov": round(cov, 4),
                    "throughput_enc_s": round(B / (p50 / 1e6), 1),
                    "cosine_vs_cpu": round(cos, 5),
                    "cert_compute_by_buffer": cert_last.get("by_buffer"),
                    "cert_non_htp_ops": cert_last.get("non_htp_ops"),
                    "cert_status": cert_last.get("status"),
                })
                print(f"  {pname}/{ph['hexagon']} L~{L} B={B}: p50={p50/1000:.1f}ms "
                      f"cos={cos:.4f} HTP={cert_last['by_buffer'].get('HTP0')} "
                      f"cpu={cert_last.get('non_htp_ops')} ({ok}/{args.procs})", flush=True)
        devices[pname]["thermal_npu_end"] = thermal_snapshot(serial)
        thermal_end = devices[pname]["thermal_npu_end"]
        if not thermal_end["valid"] or thermal_end["max_millic"] > args.thermal_max_millic:
            failures.append(f"{pname}: invalid end thermal sample")

    result = {
        "schema": "s14-cp-b-phone-bge-atlas-v2",
        "checkpoint": "B",
        "scope": "PHONE HTP0 BGE encode latency + scheduled-placement cert; phone ENERGY UNKNOWN",
        "note": "APPEND-ONLY atlas extension; frozen island_catalog.json NOT mutated. "
                "Fills the empty p50/p95/p99 on island bge_encoder_0_12 profile rows.",
        "model": "bge-small-en-v1.5-f16", "model_sha256": MODEL_SHA,
        "flash_attn": "off (explicit softmax)",
        "remote_bin": args.remote_bin,
        "bench": {"procs": args.procs, "reps_per_proc": args.reps, "warmup": args.warmup, "mbuf_mib": args.mbuf},
        "cosine_gate": args.cosine_gate, "cov_gate": args.cov_gate,
        "thermal_max_millic": args.thermal_max_millic,
        "declared_cpu_ops": sorted(DECLARED_CPU_OPS),
        "server_profile_sha256": sha256_file(args.server_profile),
        "matched_server_seq_exact": {str(k): v for k, v in server_exact.items()},
        "cpu_reference_embedding_dims": {str(L): len(v) for L, v in refs.items()},
        "devices": devices, "shapes": shapes,
        "failures": failures, "n_failures": len(failures),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nfailures={len(failures)}  wrote {args.output}", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
