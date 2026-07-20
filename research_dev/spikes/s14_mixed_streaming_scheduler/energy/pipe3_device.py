#!/usr/bin/env python3
"""S14 full 3-device pipeline: A6000 tail [k2,48) + OP15 head [0,k1) + OP12 mid [k1,k2).

Realises the [0,12)=22.8% depth ceiling that a SINGLE phone cannot hold (the ~4 GiB
HTP single-buffer cap stops one phone at [0,8)). The host `pipedriver` tail connects
to BOTH phone stagenet stages via --port (A=OP15 head) and --port2 (B=OP12 mid); the
binary's validate_stage_chain enforces the contiguous topology
  OP15 [0,k1)  ->  OP12 [k1,k2)  ->  host [k2,48).

Reuses the UNCHANGED binaries: op15/op12 run `--mode stagenet` (head-less, KV-resident,
env LLAMA_LAYER_START/END), the host runs `--mode pipedriver` (LLAMA_LAYER_START=k2).
A host full-model run at the SAME batch is the correctness reference (batched greedy
decode is not bit-identical to single-stream -- float non-associativity -- so a batched
route is 'correct' iff it matches the full model AT THE SAME BATCH; at batch 1 the
route is deterministic and must match exactly).

Scope: 3-device pipeline MECHANICS + per-device stage latency + token-correctness.
GPU-board energy is a separate stage; phone energy remains UNKNOWN.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import stageb_headcert as sb

HERE = Path(__file__).resolve().parent
HOST_BIN = sb.HOST_BIN
FULL_MODEL = sb.FULL_MODEL
N_LAYER = sb.N_LAYER

OP15_SERIAL = "3C15AU002CL00000"   # Hexagon v81, head [0,k1)
OP12_SERIAL = "5ae7a43d"            # Hexagon v75, mid  [k1,k2)
HEAD_SHARD = "/data/local/tmp/ls-npu/12b-f16-head-0-{k1}.gguf"
MID_SHARD = "/data/local/tmp/ls-npu/12b-f16-mid-{k1}-{k2}.gguf"


def start_stage(serial: str, shard: str, ls: int, le: int, port: int, batch: int,
                n_gen: int, ctx: int, max_prefill: int, mbuf: int, log_path: Path,
                remote_dir: str, remote_bin: str, timeout: int) -> tuple[subprocess.Popen, Any]:
    """Start a phone stagenet stage [ls,le) on HTP0 and wait until it is listening."""
    sb.adb(serial, "forward", "--remove", f"tcp:{port}", check=False)
    sb.adb(serial, "forward", f"tcp:{port}", f"tcp:{port}")
    env_kv = (f"LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF={mbuf} "
              f"LLAMA_LAYER_START={ls} LLAMA_LAYER_END={le} LAYERSPLIT_PLACEMENT_CERT=1")
    cmd = (f"cd {shlex.quote(remote_dir)} && env {env_kv} ./{shlex.quote(remote_bin)} "
           f"-m {shlex.quote(shard)} --devices HTP0 -ngl 99 --mode stagenet --port {port} "
           f"-n {n_gen} --driver-batch {batch} --driver-context {ctx} --driver-max-prefill {max_prefill}")
    handle = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(["adb", "-s", serial, "shell", cmd],
                            stdout=handle, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handle.flush()
        txt = log_path.read_text(encoding="utf-8", errors="replace")
        if "[stagenet] listening" in txt:
            return proc, handle
        if proc.poll() is not None:
            handle.close()
            raise RuntimeError(f"stage [{ls},{le}) on {serial} exited before listening:\n{txt[-2500:]}")
        time.sleep(0.3)
    proc.kill(); handle.close()
    raise RuntimeError(f"stage [{ls},{le}) on {serial} did not listen within {timeout}s")


def stop_stage(serial: str, proc: subprocess.Popen, handle: Any, port: int, remote_bin: str) -> None:
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        sb.adb(serial, "shell", f"pkill -9 -f {shlex.quote(remote_bin)}", check=False)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    handle.flush(); handle.close()
    sb.adb(serial, "forward", "--remove", f"tcp:{port}", check=False)


DECLARED_CPU_OPS = {"GET_ROWS"}  # tok_embd get_rows on CPU: the one declared f16-table exception


def validate_stage_cert(cert: dict | None, exp_start: int, exp_end: int) -> list[str]:
    """Fail-closed checks on a layersplit stage placement certificate. Empty list == pass."""
    if cert is None:
        return ["no PLACEMENTCERT emitted"]
    reasons = []
    if cert.get("status") != "SCHEDULED_PLACEMENT_OK":
        reasons.append(f"status={cert.get('status')}")
    if cert.get("missing_buffer_compute_nodes", 1) != 0:
        reasons.append(f"missing_buffer={cert.get('missing_buffer_compute_nodes')}")
    if cert.get("layer_start") != exp_start or cert.get("layer_end") != exp_end:
        reasons.append(f"range=[{cert.get('layer_start')},{cert.get('layer_end')}) != [{exp_start},{exp_end})")
    comp = cert.get("compute_by_buffer_type", {})
    if sum(v for k, v in comp.items() if "HTP" in k) <= 0:
        reasons.append("HTP compute nodes == 0")
    for op, bufs in (cert.get("compute_by_op_and_buffer") or {}).items():
        for buf, n in bufs.items():
            if "HTP" not in buf and "CUDA" not in buf and op not in DECLARED_CPU_OPS:
                reasons.append(f"undeclared CPU op {op}@{buf}:{n}")
    return reasons


def build_checks(host_rc: int, rows: list, certA: dict | None, certB: dict | None,
                 k1: int, k2: int, ref_tokens: list) -> dict:
    """Fail-closed acceptance checks for a 3-device run. All True == certified."""
    val_a = validate_stage_cert(certA, 0, k1)
    val_b = validate_stage_cert(certB, k1, k2)
    return {
        "host_returncode_zero": host_rc == 0,
        "result_rows_present": len(rows) >= 1,
        "op15_head_cert_ok": not val_a,
        "op12_mid_cert_ok": not val_b,
        "ranges_match_topology": (certA or {}).get("layer_end") == k1
                                 and (certB or {}).get("layer_start") == k1
                                 and (certB or {}).get("layer_end") == k2,
        "token_match_stream0": bool(rows) and rows[0].get("token_ids", []) == ref_tokens,
        "all_streams_match_reference": bool(rows) and all(r.get("token_ids", []) == ref_tokens for r in rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-uuid", default="GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf")
    ap.add_argument("--k1", type=int, default=8, help="OP15 head end / OP12 mid start")
    ap.add_argument("--k2", type=int, default=12, help="OP12 mid end / host tail start")
    ap.add_argument("--portA", type=int, default=15577)   # op15 head
    ap.add_argument("--portB", type=int, default=15578)   # op12 mid
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--n-gen", type=int, default=8)
    ap.add_argument("--requests", type=int, default=0)    # default -> batch (divisible)
    ap.add_argument("--warmups", type=int, default=0)     # default -> batch
    ap.add_argument("--driver-context", type=int, default=512)
    ap.add_argument("--driver-max-prefill", type=int, default=64)
    ap.add_argument("--prompt", default="Explain why batching improves accelerator utilization.")
    ap.add_argument("--remote-dir", default="/data/local/tmp/ls-s14")
    ap.add_argument("--remote-bin", default="llama-layersplit")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--output", default=str(HERE / "pipe3_result.json"))
    args = ap.parse_args()

    reqs = args.requests if args.requests > 0 else args.batch
    warm = args.warmups if args.warmups > 0 else args.batch
    k1, k2 = args.k1, args.k2
    head_shard = HEAD_SHARD.format(k1=k1)
    mid_shard = MID_SHARD.format(k1=k1, k2=k2)
    log_dir = HERE / "logs_pipe3"
    log_dir.mkdir(exist_ok=True)

    ref_kind = "single-stream" if args.batch == 1 else f"full-model batch={args.batch}"
    print(f"[ref] host full-model reference ({ref_kind})...", flush=True)
    ref_tokens = sb.run_mono_reference(args.gpu_uuid, args.prompt, args.n_gen, args.timeout,
                                       batch=args.batch, ctx=args.driver_context,
                                       max_prefill=args.driver_max_prefill)
    print(f"[ref] token_ids={ref_tokens}", flush=True)

    head_mbuf = sb.mbuf_for_k(k1)                       # op15 head [0,k1)
    mid_mbuf = max(3072, int((k2 - k1) * 428) + 768)    # op12 mid  [k1,k2)

    procA = handleA = procB = handleB = None
    try:
        print(f"[op15] head [0,{k1}) mbuf={head_mbuf} port={args.portA} ...", flush=True)
        procA, handleA = start_stage(OP15_SERIAL, head_shard, 0, k1, args.portA, args.batch,
                                     args.n_gen, args.driver_context, args.driver_max_prefill,
                                     head_mbuf, log_dir / "op15_head.log", args.remote_dir,
                                     args.remote_bin, args.timeout)
        print(f"[op12] mid [{k1},{k2}) mbuf={mid_mbuf} port={args.portB} ...", flush=True)
        procB, handleB = start_stage(OP12_SERIAL, mid_shard, k1, k2, args.portB, args.batch,
                                     args.n_gen, args.driver_context, args.driver_max_prefill,
                                     mid_mbuf, log_dir / "op12_mid.log", args.remote_dir,
                                     args.remote_bin, args.timeout)
        print("[host] tail pipedriver [%d,48) driving both stages ..." % k2, flush=True)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
        env["LD_LIBRARY_PATH"] = str(Path(HOST_BIN).parent) + ":" + env.get("LD_LIBRARY_PATH", "")
        env["LLAMA_LAYER_START"] = str(k2)
        env.pop("LLAMA_LAYER_END", None)
        host_cmd = [HOST_BIN, "-m", FULL_MODEL, "-ngl", "99", "--mode", "pipedriver",
                    "--host", "127.0.0.1", "--port", str(args.portA), "--port2", str(args.portB),
                    "-p", args.prompt, "-n", str(args.n_gen),
                    "--driver-requests", str(reqs), "--driver-warmup", str(warm),
                    "--driver-batch", str(args.batch), "--driver-context", str(args.driver_context),
                    "--driver-max-prefill", str(args.driver_max_prefill)]
        host = subprocess.run(host_cmd, capture_output=True, text=True, env=env, timeout=args.timeout)
        (log_dir / "host_tail.stderr").write_text(host.stderr, encoding="utf-8", errors="replace")
    finally:
        if procB is not None:
            stop_stage(OP12_SERIAL, procB, handleB, args.portB, args.remote_bin)
        if procA is not None:
            stop_stage(OP15_SERIAL, procA, handleA, args.portA, args.remote_bin)

    head_txt = (log_dir / "op15_head.log").read_text(encoding="utf-8", errors="replace")
    mid_txt = (log_dir / "op12_mid.log").read_text(encoding="utf-8", errors="replace")
    rows = sb.parse_routejson(host.stderr)
    certA = sb.parse_placement_cert(head_txt)
    certB = sb.parse_placement_cert(mid_txt)
    bufA = sb.parse_model_buffers(head_txt)
    bufB = sb.parse_model_buffers(mid_txt)

    if not rows:
        print("[FAIL] no ROUTEJSON from host tail. host stderr tail:", flush=True)
        print(host.stderr[-2000:], flush=True)
        result = {
            "schema": "s14-pipe3-device-v1", "status": "NO_ROUTE",
            "layer_split": {"op15_head": [0, k1], "op12_mid": [k1, k2], "host_tail": [k2, N_LAYER]},
            "batch": args.batch, "host_returncode": host.returncode,
            "op15_cert": certA, "op12_cert": certB,
            "op15_buffers_mib": bufA, "op12_buffers_mib": bufB,
            "host_stderr_tail": host.stderr[-2000:],
            "op12_mid_tail": mid_txt[-2000:],
        }
        Path(args.output).write_text(json.dumps(result, indent=2))
        return 1

    stage_a = sorted(r["stage_a_us"] for r in rows if "stage_a_us" in r)   # op15 head
    stage_b = sorted(r.get("stage_b_us", 0) for r in rows if "stage_b_us" in r)  # op12 mid (if reported)
    host_us = sorted(r["host_us"] for r in rows if "host_us" in r)
    wall = sorted(r["request_wall_us"] for r in rows if "request_wall_us" in r)
    route_tokens = rows[0].get("token_ids", [])
    token_match = route_tokens == ref_tokens
    # every measured stream must match the same-batch correctness reference
    all_streams_match = all(r.get("token_ids", []) == ref_tokens for r in rows)

    val_a = validate_stage_cert(certA, 0, k1)
    val_b = validate_stage_cert(certB, k1, k2)
    checks = build_checks(host.returncode, rows, certA, certB, k1, k2, ref_tokens)
    certified = all(checks.values())

    def p50(x): return x[len(x) // 2] if x else None
    result = {
        "schema": "s14-pipe3-device-v2",
        "status": "CERTIFIED_3DEVICE_TOKEN_CORRECT_MECHANICS" if certified else "FAIL",
        "certified": certified,
        "label": "token-correct 3-device MECHANICS only; no latency/energy/throughput claim",
        "checks": checks,
        "cert_reasons_op15": val_a, "cert_reasons_op12": val_b,
        "scope": "3DEVICE_PIPELINE_MECHANICS_TOKEN_CORRECTNESS; GPU_BOARD_ENERGY_SEPARATE; PHONE_ENERGY_UNKNOWN",
        "layer_split": {"op15_head": [0, k1], "op12_mid": [k1, k2], "host_tail": [k2, N_LAYER],
                        "depth_offloaded_frac": round(k2 / N_LAYER, 4)},
        "route": rows[0].get("route", "A0_OP15_OP12"),
        "batch": args.batch, "n_gen": args.n_gen, "n_requests_measured": len(rows),
        "ref_kind": ref_kind, "ref_token_ids": ref_tokens, "route_token_ids": route_tokens,
        "token_match": token_match,
        "op15_head": {
            "placement_status": (certA or {}).get("status"),
            "missing_buffer_compute_nodes": (certA or {}).get("missing_buffer_compute_nodes"),
            "htp0_weight_mib": bufA.get("HTP0", 0.0),
            "stage_a_us_p50": p50(stage_a), "hexagon": "v81",
        },
        "op12_mid": {
            "placement_status": (certB or {}).get("status"),
            "missing_buffer_compute_nodes": (certB or {}).get("missing_buffer_compute_nodes"),
            "htp0_weight_mib": bufB.get("HTP0", 0.0),
            "stage_b_us_p50": p50(stage_b) if stage_b else None, "hexagon": "v75",
        },
        "host_tail": {"host_us_p50": p50(host_us)},
        "request_wall_us_p50": p50(wall),
        "op15_cert": certA, "op12_cert": certB,
        "op15_buffers_mib": bufA, "op12_buffers_mib": bufB,
        "host_returncode": host.returncode,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\n[result] certified={certified}", flush=True)
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    if val_a:
        print(f"    op15 cert reasons: {val_a}", flush=True)
    if val_b:
        print(f"    op12 cert reasons: {val_b}", flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0 if certified else 2


if __name__ == "__main__":
    sys.exit(main())
