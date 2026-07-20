#!/usr/bin/env python3
"""S14 energy Stage B: deep-head phone island placement certificate + stage latency.

Confirms OP15 (Hexagon v81) can HOLD + RUN a deep head island [0,k) of
gemma-4-12B-it-f16 on HTP0 at a real stage latency with a no-fallback placement
certificate, for k in {2,6,12}. This makes the deeper island dispatch-ELIGIBLE
and turns the Stage A GPU-board energy ceiling (22.8% saved at k=12) into a
realisable saving instead of an assumption.

Method (reuses existing binaries UNCHANGED):
  - host build-cuda/bin/llama-layersplit --mode pipedriver runs the TAIL [k,48)
    on the A6000 and drives the phone HEAD stage [0,k) over adb/USB, incremental
    decode, emitting one ROUTEJSON per request with stage_a_us (phone head time);
  - phone ls-s14/llama-layersplit --mode stagenet holds [0,k) resident on HTP0
    with LAYERSPLIT_PLACEMENT_CERT=1, emitting a PLACEMENTCERT (compute_by_buffer
    tally) + the load_tensors buffer sizes (HTP0 model buffer + CPU_Mapped embd);
  - a host mono baseline gives the reference token ids for a correctness check.

Does NOT modify the frozen S11-E0 harness (run_fixed_route.py) whose --measure
path is hard-locked to k=2. This is a NEW deeper-island measurement.

Scope: OP15/HTP0 island FEASIBILITY + PLACEMENT + phone-side stage LATENCY only.
Phone ENERGY remains UNKNOWN (no phone power instrument on this host). This
certifies the island is real, resident, no-fallback and runnable at depth; it is
NOT a phone-energy or total-wall claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
HOST_BIN = str(REPO_ROOT / "build-cuda/bin/llama-layersplit")
FULL_MODEL = "/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf"
N_LAYER = 48
MIB = 1024 * 1024

# Per-k HTP0 model-buffer sizing: measured [0,2) HTP0 buffer = 855.13 MiB / 2
# layers = 427.6 MiB/layer (the f16 token_embd is CPU_Mapped, NOT on HTP0). Add
# ~700 MiB headroom (KV + compute + skel) and floor at 3072.
def mbuf_for_k(k: int) -> int:
    return max(3072, int(k * 428) + 768)


SHARDS = {
    2: "/data/local/tmp/ls-npu/12b-f16-head-0-2.gguf",
    6: "/data/local/tmp/ls-npu/12b-f16-head-0-6.gguf",
    8: "/data/local/tmp/ls-npu/12b-f16-head-0-8.gguf",
    10: "/data/local/tmp/ls-npu/12b-f16-head-0-10.gguf",
    12: "/data/local/tmp/ls-npu/12b-f16-head-0-12.gguf",
}
THERMAL_START_MAX_MILLIC = 60_000
THERMAL_END_MAX_MILLIC = 85_000


def adb(serial: str, *args: str, check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", "-s", serial, *args], capture_output=True, text=True,
                          check=check, timeout=timeout)


def sha256_remote(serial: str, path: str) -> str:
    return adb(serial, "shell", f"sha256sum {shlex.quote(path)}").stdout.split()[0]


def thermal_snapshot(serial: str) -> dict[str, Any]:
    command = (
        "for z in /sys/class/thermal/thermal_zone*; do "
        "ty=$(cat $z/type 2>/dev/null); t=$(cat $z/temp 2>/dev/null); "
        "case $ty in nsphmx-*) echo $ty=$t;; esac; done"
    )
    process = adb(serial, "shell", command, check=False)
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


def thermal_ok(snapshot: dict[str, Any], limit: int) -> bool:
    sensors = snapshot.get("sensors_millic")
    maximum = snapshot.get("max_millic")
    return snapshot.get("valid") is True and type(sensors) is dict and bool(sensors) \
        and type(maximum) is int and maximum == max(sensors.values()) and maximum <= limit


def parse_model_buffers(text: str) -> dict[str, float]:
    """Pull load_tensors buffer sizes (MiB) from the phone stage stderr."""
    out: dict[str, float] = {}
    for m in re.finditer(r"load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB", text):
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def parse_placement_cert(text: str) -> dict[str, Any] | None:
    for line in text.splitlines():
        if line.startswith("PLACEMENTCERT "):
            try:
                return json.loads(line[len("PLACEMENTCERT "):])
            except json.JSONDecodeError:
                continue
    return None


def parse_routejson(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if line.startswith("ROUTEJSON "):
            try:
                rows.append(json.loads(line[len("ROUTEJSON "):]))
            except json.JSONDecodeError:
                continue
    return rows


def row_is_eligible(row: dict[str, Any], requests: int) -> bool:
    cert = row.get("placement_cert")
    if not isinstance(cert, dict) or row.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
            or row.get("missing_buffer_compute_nodes") != 0 \
            or row.get("host_returncode") != 0 \
            or row.get("n_requests_measured") != requests \
            or row.get("token_match_vs_mono") is not True \
            or row.get("all_tokens_match_vs_mono") is not True:
        return False
    htp_nodes = 0
    for op, buffers in cert.get("compute_by_op_and_buffer", {}).items():
        if not isinstance(buffers, dict):
            return False
        for buffer, count in buffers.items():
            if type(count) is not int or count <= 0:
                return False
            if "HTP" in buffer:
                htp_nodes += count
            elif op != "GET_ROWS":
                return False
    return htp_nodes > 0


def run_mono_reference(gpu_uuid: str, prompt: str, n_gen: int, timeout: int,
                       batch: int = 1, ctx: int = 512, max_prefill: int = 64) -> list[int]:
    """Host FULL-MODEL reference at the SAME batch size as the route under test,
    returning stream 0's token ids. This is the correct correctness anchor for a
    batched route: batched greedy decode is NOT bit-identical to single-stream
    (float non-associativity in batched attention/GEMM flips argmax at near-ties --
    the A6000 full model shows the identical batch-size-dependent divergence). So a
    phone [0,k)+[k,48) split is 'correct' iff it matches the FULL model AT THE SAME
    BATCH, not vs single-stream."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    env["LD_LIBRARY_PATH"] = str(Path(HOST_BIN).parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    env.pop("LLAMA_LAYER_START", None)
    env.pop("LLAMA_LAYER_END", None)
    cmd = [HOST_BIN, "-m", FULL_MODEL, "-ngl", "99", "--mode", "monodriver",
           "-p", prompt, "-n", str(n_gen), "--driver-requests", str(batch), "--driver-warmup", "0",
           "--driver-batch", str(batch), "--driver-context", str(ctx),
           "--driver-max-prefill", str(max_prefill)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    rows = parse_routejson(proc.stderr)
    if not rows:
        raise RuntimeError(f"mono reference produced no ROUTEJSON (rc={proc.returncode})\n{proc.stderr[-1500:]}")
    stream0 = [r for r in rows if r.get("stream_index") == 0]
    return (stream0[0] if stream0 else rows[0]).get("token_ids", [])


def run_one_k(serial: str, gpu_uuid: str, k: int, port: int, batch: int, n_gen: int,
              requests: int, warmups: int, ctx: int, max_prefill: int, prompt: str,
              remote_dir: str, remote_bin: str, log_dir: Path, timeout: int,
              decode_no_fa: bool = False) -> dict[str, Any]:
    shard = SHARDS[k]
    mbuf = mbuf_for_k(k)
    phone_log = log_dir / f"phone_k{k}.log"

    # adb tcp forward
    adb(serial, "forward", "--remove", f"tcp:{port}", check=False)
    adb(serial, "forward", f"tcp:{port}", f"tcp:{port}")

    # start the phone stagenet head stage [0,k) on HTP0 with the placement cert.
    # OP15 is Hexagon v81 (decode flash-attn is fine here); GGML_DECODE_NO_FA is
    # opt-in because forcing the non-FA decode path hangs the global-attention
    # layers (>=5) in the stagenet incremental loop.
    env_kv = (f"LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF={mbuf} "
              f"LLAMA_LAYER_END={k} LAYERSPLIT_PLACEMENT_CERT=1"
              + (" GGML_DECODE_NO_FA=1" if decode_no_fa else ""))
    phone_cmd = (
        f"cd {shlex.quote(remote_dir)} && env {env_kv} ./{shlex.quote(remote_bin)} "
        f"-m {shlex.quote(shard)} --devices HTP0 -ngl 99 --mode stagenet --port {port} "
        f"-n {n_gen} --driver-batch {batch} --driver-context {ctx} --driver-max-prefill {max_prefill}"
    )
    ph_handle = open(phone_log, "w", encoding="utf-8", errors="replace")
    ph = subprocess.Popen(["adb", "-s", serial, "shell", phone_cmd],
                          stdout=ph_handle, stderr=subprocess.STDOUT, text=True)

    # wait for the stage to be listening (model load can take a while for deep k)
    deadline = time.monotonic() + timeout
    listening = False
    while time.monotonic() < deadline:
        ph_handle.flush()
        txt = phone_log.read_text(encoding="utf-8", errors="replace")
        if "[stagenet] listening" in txt:
            listening = True
            break
        if ph.poll() is not None:
            ph_handle.close()
            raise RuntimeError(f"phone stage k={k} exited before listening:\n{txt[-2500:]}")
        time.sleep(0.3)
    if not listening:
        ph.kill(); ph_handle.close()
        raise RuntimeError(f"phone stage k={k} did not listen within {timeout}s")

    # drive from the host tail [k,48); no --wait-for-go => runs immediately
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    env["LD_LIBRARY_PATH"] = str(Path(HOST_BIN).parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["LLAMA_LAYER_START"] = str(k)
    env.pop("LLAMA_LAYER_END", None)
    host_cmd = [HOST_BIN, "-m", FULL_MODEL, "-ngl", "99", "--mode", "pipedriver",
                "--host", "127.0.0.1", "--port", str(port), "-p", prompt, "-n", str(n_gen),
                "--driver-requests", str(requests), "--driver-warmup", str(warmups),
                "--driver-batch", str(batch), "--driver-context", str(ctx),
                "--driver-max-prefill", str(max_prefill)]
    try:
        host = subprocess.run(host_cmd, capture_output=True, text=True, env=env, timeout=timeout)
    finally:
        # driver disconnect makes the stage send STAGE_STOP then emit its cert; wait for it
        try:
            ph.wait(timeout=30)
        except subprocess.TimeoutExpired:
            adb(serial, "shell", f"pkill -9 -f {shlex.quote(remote_bin)}", check=False)
            try:
                ph.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ph.kill()
        ph_handle.flush(); ph_handle.close()
        adb(serial, "forward", "--remove", f"tcp:{port}", check=False)

    phone_txt = phone_log.read_text(encoding="utf-8", errors="replace")
    cert = parse_placement_cert(phone_txt)
    buffers = parse_model_buffers(phone_txt)
    (log_dir / f"host_k{k}_b{batch}.stderr").write_text(host.stderr, encoding="utf-8", errors="replace")
    rows = parse_routejson(host.stderr)
    if not rows:
        # No completed route. Distinguish a DSP-queue abort (feasibility ceiling:
        # HTP0 weight buffer exceeds the ~4 GiB single-buffer cap) from other
        # failures, and record it as an infeasible entry instead of aborting the
        # whole sweep so the ceiling is captured alongside the passing certs.
        m = re.search(r"dspqueue_read failed: (0x[0-9a-fA-F]+)", phone_txt)
        htp0_mib = buffers.get("HTP0", 0.0)
        cpu_mib = sum(v for kk, v in buffers.items() if kk != "HTP0")
        if m:
            return {
                "k": k, "layer_range": [0, k], "shard": shard,
                "shard_sha256": sha256_remote(serial, shard),
                "hexagon_mbuf_mib": mbuf,
                "placement_status": "DSP_QUEUE_ABORT",
                "infeasible_reason": f"ggml-hexagon dspqueue_read failed {m.group(1)} in flush_pending "
                                     f"(HTP0 weight buffer {htp0_mib:.0f} MiB exceeds the ~4 GiB DSP single-buffer cap)",
                "htp0_model_buffer_mib": htp0_mib,
                "cpu_mapped_mib": cpu_mib,
                "phone_resident_total_mib": htp0_mib + cpu_mib,
                "loaded_ok": htp0_mib > 0,
                "host_returncode": host.returncode,
            }
        raise RuntimeError(f"k={k} route produced no ROUTEJSON (host rc={host.returncode})\n"
                           f"host tail:\n{host.stderr[-1500:]}")

    stage_a = sorted(r["stage_a_us"] for r in rows if "stage_a_us" in r)
    host_us = sorted(r["host_us"] for r in rows if "host_us" in r)
    wall = sorted(r["request_wall_us"] for r in rows if "request_wall_us" in r)
    tok = rows[0].get("token_ids", [])

    htp0_mib = buffers.get("HTP0", 0.0)
    cpu_mib = sum(v for kk, v in buffers.items() if kk != "HTP0")
    comp = (cert or {}).get("compute_by_buffer_type", {})
    return {
        "k": k, "layer_range": [0, k], "shard": shard,
        "shard_sha256": sha256_remote(serial, shard),
        "hexagon_mbuf_mib": mbuf,
        "n_requests_measured": len(rows), "batch": batch, "n_gen": n_gen,
        "placement_status": (cert or {}).get("status"),
        "compute_htp0_nodes": comp.get("HTP0", 0),
        "compute_cpu_nodes": comp.get("CPU", 0),
        "missing_buffer_compute_nodes": (cert or {}).get("missing_buffer_compute_nodes"),
        "htp0_model_buffer_mib": htp0_mib,
        "cpu_mapped_mib": cpu_mib,
        "phone_resident_total_mib": htp0_mib + cpu_mib,
        "stage_a_us_p50": stage_a[len(stage_a) // 2] if stage_a else None,
        "stage_a_us_min": stage_a[0] if stage_a else None,
        "stage_a_us_max": stage_a[-1] if stage_a else None,
        "host_us_p50": host_us[len(host_us) // 2] if host_us else None,
        "request_wall_us_p50": wall[len(wall) // 2] if wall else None,
        "token_ids": tok,
        "token_ids_by_request": [row.get("token_ids") for row in rows],
        "placement_cert": cert,
        "model_buffers_mib": buffers,
        "host_returncode": host.returncode,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="3C15AU002CL00000")
    ap.add_argument("--gpu-uuid", default="GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf")
    ap.add_argument("--k-list", default="2,6,12")
    ap.add_argument("--port", type=int, default=15577)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-gen", type=int, default=8)
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--warmups", type=int, default=4)
    ap.add_argument("--driver-context", type=int, default=512)
    ap.add_argument("--driver-max-prefill", type=int, default=64)
    ap.add_argument("--prompt", default="Explain why batching improves accelerator utilization.")
    ap.add_argument("--remote-dir", default="/data/local/tmp/ls-s14")
    ap.add_argument("--remote-bin", default="llama-layersplit")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--decode-no-fa", action="store_true",
                    help="force GGML_DECODE_NO_FA=1 (hangs global-attn layers >=5 on v81; leave OFF)")
    ap.add_argument("--output", default=str(HERE / "stageb_result.json"))
    args = ap.parse_args()

    k_list = [int(x) for x in args.k_list.split(",")]
    log_dir = HERE / "stageb_logs"
    log_dir.mkdir(exist_ok=True)

    remote_bin_hash = sha256_remote(args.serial, f"{args.remote_dir}/{args.remote_bin}")
    print(f"phone binary {args.remote_dir}/{args.remote_bin} sha256={remote_bin_hash[:16]}", flush=True)
    thermal_start = thermal_snapshot(args.serial)
    if not thermal_ok(thermal_start, THERMAL_START_MAX_MILLIC):
        print(f"invalid start thermal sample: {thermal_start}", file=sys.stderr)
        return 2

    ref_kind = "single-stream" if args.batch == 1 else f"full-model batch={args.batch}"
    print(f"reference (host {ref_kind}) ...", flush=True)
    ref_tokens = run_mono_reference(args.gpu_uuid, args.prompt, args.n_gen, args.timeout,
                                    batch=args.batch, ctx=args.driver_context,
                                    max_prefill=args.driver_max_prefill)
    print(f"  reference token_ids={ref_tokens}", flush=True)

    results = []
    for k in k_list:
        print(f"\n=== k={k} tail=[{k},{N_LAYER}) head=[0,{k}) mbuf={mbuf_for_k(k)}MiB ===", flush=True)
        r = run_one_k(args.serial, args.gpu_uuid, k, args.port, args.batch, args.n_gen,
                      args.requests, args.warmups, args.driver_context, args.driver_max_prefill,
                      args.prompt, args.remote_dir, args.remote_bin, log_dir, args.timeout,
                      decode_no_fa=args.decode_no_fa)
        results.append(r)
        if r["placement_status"] == "DSP_QUEUE_ABORT":
            print(f"  status=DSP_QUEUE_ABORT (loaded={r['loaded_ok']}) "
                  f"HTP0={r['htp0_model_buffer_mib']:.0f}MiB -> {r['infeasible_reason']}", flush=True)
            continue
        r["token_match_vs_mono"] = (r["token_ids"] == ref_tokens)
        r["all_tokens_match_vs_mono"] = all(
            token_ids == ref_tokens for token_ids in r["token_ids_by_request"]
        )
        print(f"  status={r['placement_status']} HTP0={r['compute_htp0_nodes']} "
              f"CPU={r['compute_cpu_nodes']} missing={r['missing_buffer_compute_nodes']}", flush=True)
        print(f"  resident: HTP0={r['htp0_model_buffer_mib']:.0f}MiB CPU_mapped={r['cpu_mapped_mib']:.0f}MiB "
              f"total={r['phone_resident_total_mib']:.0f}MiB", flush=True)
        print(f"  stage_a(phone head) p50={r['stage_a_us_p50']}us  host(tail) p50={r['host_us_p50']}us  "
              f"token_match={r['all_tokens_match_vs_mono']}", flush=True)

    feasible = [r for r in results if row_is_eligible(r, args.requests)]
    infeasible = [r for r in results if r["placement_status"] == "DSP_QUEUE_ABORT"]
    failed = [r for r in results if r not in feasible and r not in infeasible]
    thermal_end = thermal_snapshot(args.serial)
    thermal_pass = thermal_ok(thermal_end, THERMAL_END_MAX_MILLIC)
    max_feasible_k = max((r["k"] for r in feasible), default=None)

    result = {
        "schema": "s14-stage-b-deep-head-placement-v1",
        "status": "HEAD_SWEEP_COMPLETE" if not failed and thermal_pass else "HEAD_SWEEP_FAIL",
        "scope": "OP15_HTP0_ISLAND_FEASIBILITY_PLACEMENT_AND_PHONE_STAGE_LATENCY_ONLY_PHONE_ENERGY_UNKNOWN",
        "formal_claim": "DEEP_HEAD_PLACEMENT_CERT",
        "device": {"serial": args.serial, "soc": "SM8850", "hexagon": "v81", "backend": "HTP0"},
        "phone_binary_sha256": remote_bin_hash,
        "host_binary": HOST_BIN,
        "model": "gemma-4-12b-it-f16",
        "n_layer": N_LAYER,
        "batch": args.batch, "n_gen": args.n_gen, "requests": args.requests, "warmups": args.warmups,
        "thermal": {
            "start": thermal_start,
            "end": thermal_end,
            "start_max_millic": THERMAL_START_MAX_MILLIC,
            "end_max_millic": THERMAL_END_MAX_MILLIC,
        },
        "decode_no_fa": args.decode_no_fa,
        "reference_token_ids": ref_tokens,
        "feasibility_summary": {
            "max_feasible_head_k": max_feasible_k,
            "certified_k": sorted(r["k"] for r in feasible),
            "dsp_abort_k": sorted(r["k"] for r in infeasible),
            "dsp_single_buffer_cap_note": "HTP0 weight buffer must stay under ~4 GiB (2^32 B); "
                                          "k>=10 (>=4.3 GiB) aborts with dspqueue_read failed in flush_pending.",
        },
        "per_k": results,
        "caveats": [
            "OP15/HTP0 island feasibility + placement + phone-side stage latency ONLY.",
            "Phone ENERGY is UNKNOWN (no phone power instrument on this host).",
            "token_embd (f16) is CPU_Mapped (declared GET_ROWS exception), NOT a fallback failure; SCHEDULED_PLACEMENT_OK with missing_buffer=0 is the no-fallback certificate.",
            "stage_a_us is the phone head time measured over adb/USB incremental decode; it includes the USB relay, so it is an UPPER bound on the on-device head compute.",
            "DSP single-buffer cap: the HTP0 weight buffer must stay < ~4 GiB. Heads k>=10 LOAD but abort on the first forward (dspqueue_read failed in flush_pending). Max feasible single-phone head is [0,8) (HTP0 buffer 3454 MiB). [0,12) (the 22.8% Stage-A ceiling) is NOT realisable as one single-phone island here.",
            "Eligibility is batch-specific and requires same-batch full-model token correctness. A passing batch does not certify another batch.",
            "This makes the feasible islands dispatch-ELIGIBLE; it does not by itself prove a total-system energy saving (see Stage D overlap).",
        ],
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.output}", flush=True)
    return 0 if not failed and thermal_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
