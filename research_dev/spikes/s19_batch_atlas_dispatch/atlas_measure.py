#!/usr/bin/env python3
"""S19 CP1 batch-atlas measurement harness (real devices, no energy).

Drives the existing persistent LayerSplit drivers (no C++ change):
  - CUDA_R0 : --mode monodriver --persistent-jsonl (full model on the selected A6000)
  - OP15_R1 : resident stagenet [0,8) worker + host pipedriver --persistent-jsonl
  - OP12_R1 : resident stagenet [0,6) worker + host pipedriver --persistent-jsonl

Each atlas cell is one (route, batch). A cell aggregates measured DETACH-reset
exchanges over `replicates` fresh host processes (each a 23GB model reload) with
`warmup`+`measured` exchanges per process (DETACH between, STOP last). See
PLAN.md sections 6-7 for the frozen protocol and the declared "process" caveat.

Outputs are appended incrementally to results/atlas_rows.jsonl and raw logs to
raw/. Deterministic identities are pinned in PLAN.md section 3.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/home/myid/zs89458/Documents/llama.cpp-release")
HERE = Path(__file__).resolve().parent
HOST_BIN = ROOT / "build-cuda/bin/llama-layersplit"
HOST_LIB = HOST_BIN.parent
FROZEN_HOST = HERE / "artifacts/llama-layersplit-host-cuda"
FULL_MODEL = Path("/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf")

SELECTED_GPU = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"
SECOND_GPU = "GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf"

PROMPT = "Explain batching."
PROMPT_SHA = "c544378e66bf7a3d6640824b9e4cc28deaef2e35a424772b56b95d27c17fe6c8"
N_GEN = 8
CONTEXT = 16
MAX_PREFILL = 8
REMOTE = "/data/local/tmp/ls-s14-persistent"

FULL_MODEL_HASH = "bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a"
HOST_BIN_HASH = "da12f9255d2e7acf276c3803cbfab2e7e2aaed1ed50230a43bd8bd1f79f2c8f7"
PHONE_BIN_HASH = "d26075bcf64e90ee04e51c2f86188d709e4c2906add252d209014ad1e046646c"

ROUTES = {
    "CUDA_R0": {
        "kind": "cuda", "host_layer_start": 0, "layer_range": [0, 48],
        "backend": "CUDA0", "device": SELECTED_GPU,
    },
    "OP15_R1": {
        "kind": "phone", "serial": "3C15AU002CL00000", "port": 5991, "layer_end": 8,
        "shard": "/data/local/tmp/ls-npu/12b-f16-head-0-8.gguf", "mbuf": 4192,
        "host_layer_start": 8, "layer_range": [0, 8], "backend": "HTP0",
        "device": "3C15AU002CL00000",
        "head_hash": "a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8",
    },
    "OP12_R1": {
        "kind": "phone", "serial": "5ae7a43d", "port": 5992, "layer_end": 6,
        "shard": "/data/local/tmp/ls-npu/12b-f16-head-0-6.gguf", "mbuf": 3336,
        "host_layer_start": 6, "layer_range": [0, 6], "backend": "HTP0",
        "device": "5ae7a43d",
        "head_hash": "d507b7bb453242dff12ba1ce0add53189755b8a9a2b960743ead1578d8f1a6b5",
    },
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


# ---------------------------------------------------------------------------
# nvidia-smi sampling
# ---------------------------------------------------------------------------
def nvml_snapshot() -> dict[str, dict[str, int]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30).stdout
    snap: dict[str, dict[str, int]] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            snap[parts[0]] = {"mem_mib": int(parts[1]), "util": int(parts[2])}
    return snap


class GpuPeakSampler:
    def __init__(self, uuid: str) -> None:
        self.uuid = uuid
        self.peak = 0
        self.gpu1_util_max = 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                snap = nvml_snapshot()
                if self.uuid in snap:
                    self.peak = max(self.peak, snap[self.uuid]["mem_mib"])
                if SECOND_GPU in snap:
                    self.gpu1_util_max = max(self.gpu1_util_max, snap[SECOND_GPU]["util"])
            except Exception:
                pass
            self._stop.wait(0.1)

    def start(self) -> None:
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)


# ---------------------------------------------------------------------------
# phone helpers
# ---------------------------------------------------------------------------
def adb(serial: str, *args: str, timeout: int = 60, check: bool = True) -> str:
    r = subprocess.run(["adb", "-s", serial, *args], capture_output=True,
                       text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def phone_thermal_c(serial: str) -> float | None:
    """Max of the Hexagon NSP (HVX/HMX) thermal zones, in Celsius."""
    try:
        script = (
            'for z in /sys/class/thermal/thermal_zone*/type; do '
            't=$(cat ${z%type}temp 2>/dev/null); echo "$(cat $z)=$t"; done')
        out = adb(serial, "shell", script, timeout=30, check=False)
        best = None
        for line in out.splitlines():
            if "=" not in line:
                continue
            name, _, val = line.partition("=")
            if name.startswith("nsp") and val.strip().lstrip("-").isdigit():
                c = int(val.strip()) / 1000.0
                best = c if best is None else max(best, c)
        return best
    except Exception:
        return None


def kill_worker(route: dict) -> None:
    serial, port = route["serial"], route["port"]
    adb(serial, "shell", f"pkill -9 -f 'stagenet --port {port}' >/dev/null 2>&1 || true",
        check=False)
    adb(serial, "forward", "--remove", f"tcp:{port}", check=False)


# ---------------------------------------------------------------------------
# monitored subprocess with line readers
# ---------------------------------------------------------------------------
class Proc:
    def __init__(self, cmd, env=None, stdin=False):
        self.cmd = cmd
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self._lock = threading.Lock()
        self.p = subprocess.Popen(
            cmd, env=env,
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self._to = threading.Thread(target=self._reader, args=(self.p.stdout, self.stdout_lines), daemon=True)
        self._te = threading.Thread(target=self._reader, args=(self.p.stderr, self.stderr_lines), daemon=True)
        self._to.start()
        self._te.start()

    def _reader(self, stream, sink):
        for line in stream:
            with self._lock:
                sink.append(line.rstrip("\n"))

    def wait_line(self, which: str, prefix: str, timeout: float) -> str:
        deadline = time.time() + timeout
        sink = self.stdout_lines if which == "stdout" else self.stderr_lines
        while time.time() < deadline:
            with self._lock:
                for line in sink:
                    if prefix in line:
                        return line
            if self.p.poll() is not None:
                # process exited; final scan
                time.sleep(0.2)
                with self._lock:
                    for line in sink:
                        if prefix in line:
                            return line
                raise RuntimeError(f"process exited before '{prefix}' (rc={self.p.returncode})")
            time.sleep(0.05)
        raise TimeoutError(f"timeout waiting for '{prefix}' on {which}")

    def send(self, obj: dict) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def snapshot_len(self, which: str) -> int:
        with self._lock:
            return len(self.stdout_lines if which == "stdout" else self.stderr_lines)

    def lines_since(self, which: str, start: int) -> list[str]:
        with self._lock:
            sink = self.stdout_lines if which == "stdout" else self.stderr_lines
            return list(sink[start:])

    def close(self, timeout: float = 30) -> int:
        try:
            if self.p.stdin and not self.p.stdin.closed:
                self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait(timeout=5)
        return self.p.returncode


# ---------------------------------------------------------------------------
# certificate parsing
# ---------------------------------------------------------------------------
def parse_cert(lines: list[str], marker: str) -> list[dict]:
    out = []
    for line in lines:
        i = line.find(marker + " ")
        if i >= 0:
            frag = line[i + len(marker) + 1:]
            try:
                out.append(json.loads(frag))
            except Exception:
                pass
    return out


def placement_ok(cert: dict) -> bool:
    if cert.get("status") not in ("SCHEDULED_PLACEMENT_OK",) and \
       cert.get("placement_status") not in ("SCHEDULED_PLACEMENT_OK",):
        return False
    return int(cert.get("missing_buffer_compute_nodes", 1)) == 0


def cpu_ops(cert: dict) -> list[str]:
    """Ops that ran on a CPU/Host buffer, from compute_by_op_and_buffer."""
    m = cert.get("compute_by_op_and_buffer") or {}
    cpu = []
    for op, by_buf in m.items():
        for buf, _ in (by_buf or {}).items():
            if "CPU" in buf or "Host" in buf.replace("CUDA_Host", "Host"):
                cpu.append(op)
    return sorted(set(cpu))


# ---------------------------------------------------------------------------
# one host process run: warmup + measured exchanges, DETACH between, STOP last
# ---------------------------------------------------------------------------
def run_host_process(route_name, route, batch, warmup, measured, raw_prefix):
    """Returns dict with per-exchange results, host placement certs, host stderr."""
    kind = route["kind"]
    total = warmup + measured
    if kind == "cuda":
        cmd = [str(HOST_BIN), "-m", str(FULL_MODEL), "-ngl", "99",
               "--mode", "monodriver", "-n", str(N_GEN),
               "--driver-batch", str(batch), "--driver-context", str(CONTEXT),
               "--driver-max-prefill", str(MAX_PREFILL), "--persistent-jsonl"]
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": SELECTED_GPU,
                    "LD_LIBRARY_PATH": str(HOST_LIB),
                    "LAYERSPLIT_PLACEMENT_CERT": "1"})
    else:
        cmd = [str(HOST_BIN), "-m", str(FULL_MODEL), "-ngl", "99",
               "--mode", "pipedriver", "--host", "127.0.0.1",
               "--port", str(route["port"]), "-n", str(N_GEN),
               "--driver-batch", str(batch), "--driver-context", str(CONTEXT),
               "--driver-max-prefill", str(MAX_PREFILL), "--persistent-jsonl"]
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": SELECTED_GPU,
                    "LD_LIBRARY_PATH": str(HOST_LIB),
                    "LLAMA_LAYER_START": str(route["layer_end"]),
                    "LAYERSPLIT_PLACEMENT_CERT": "1"})

    host = Proc(cmd, env=env, stdin=True)
    results = []
    try:
        ready = host.wait_line("stderr", "PERSISTENT_DRIVER_READY", 420)
        _ = json.loads(ready[ready.find("{"):])
        for i in range(1, total + 1):
            session_end = "STOP" if i == total else "DETACH"
            out0 = host.snapshot_len("stdout")
            host.send({"schema": "layersplit-persistent-command-v1",
                       "launch_id": i, "prompt": PROMPT, "n_gen": N_GEN,
                       "request_count": batch, "session_end": session_end})
            # wait for the result JSONL line on stdout. A real exchange is well
            # under 20 s even at B64; a longer wait means an OP12/v75 HTP hang.
            deadline = time.time() + 90
            result = None
            while time.time() < deadline:
                for line in host.lines_since("stdout", out0):
                    line = line.strip()
                    if line.startswith("{") and '"layersplit-persistent-result-v1"' in line:
                        result = json.loads(line)
                        break
                if result is not None:
                    break
                if host.p.poll() is not None and i != total:
                    break
                time.sleep(0.03)
            if result is None:
                raise RuntimeError(f"no result for exchange {i}")
            results.append({"exchange": i, "warmup": i <= warmup,
                            "session_end": session_end, "result": result})
    finally:
        rc = host.close(timeout=60)
    host_certs = parse_cert(host.stderr_lines, "PLACEMENTCERT")
    # persist raw
    (raw_prefix.parent).mkdir(parents=True, exist_ok=True)
    raw_prefix.with_suffix(".host.stderr.log").write_text("\n".join(host.stderr_lines))
    raw_prefix.with_suffix(".host.stdout.log").write_text("\n".join(host.stdout_lines))
    return {"rc": rc, "results": results, "host_certs": host_certs,
            "host_stderr": host.stderr_lines}


def launch_phone_worker(route, batch, raw_prefix):
    serial, port = route["serial"], route["port"]
    kill_worker(route)
    time.sleep(1.0)
    adb(serial, "forward", f"tcp:{port}", f"tcp:{port}")
    shell = (
        f"cd {REMOTE} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
        f"GGML_HEXAGON_MBUF={route['mbuf']} LLAMA_LAYER_END={route['layer_end']} "
        "LAYERSPLIT_PLACEMENT_CERT=1 "
        f"./llama-layersplit -m {route['shard']} --devices HTP0 -ngl 99 "
        f"--mode stagenet --port {port} -n {N_GEN} "
        f"--driver-batch {batch} --driver-context {CONTEXT} "
        f"--driver-max-prefill {MAX_PREFILL}")
    worker = Proc(["adb", "-s", serial, "shell", shell])
    worker.wait_line("stderr", "[stagenet] listening", 300)
    # the phone worker log arrives on the adb shell's stdout OR stderr; scan both
    time.sleep(0.5)
    pid_out = adb(serial, "shell", f"pgrep -f 'stagenet --port {port}'", check=False).strip()
    worker_pid = pid_out.splitlines()[0].strip() if pid_out else None
    return worker, worker_pid


def measure_cell(route_name, batch, warmup, measured, replicates, out_dir):
    route = ROUTES[route_name]
    cell_id = f"{route_name}.B{batch}"
    log(f"--- measuring {cell_id} (warmup={warmup} measured={measured} reps={replicates}) ---")
    gpu_before = nvml_snapshot()
    thermal_start = phone_thermal_c(route["serial"]) if route["kind"] == "phone" else None

    all_walls: list[int] = []
    all_tokens: list[list[int]] = []
    host_placement_ok = True
    cpu_get_rows_only = True
    compute_nodes_total = 0
    session_certs: list[dict] = []
    worker_pids: set = set()
    worker_nonces: set = set()
    reset_flags: list[bool] = []
    rc_all = 0
    support = True
    oom = False
    err_reason = None

    sampler = GpuPeakSampler(SELECTED_GPU)
    sampler.start()
    try:
        for rep in range(replicates):
            raw_prefix = out_dir / "raw" / f"{cell_id}.rep{rep}"
            worker = None
            worker_pid = None
            try:
                if route["kind"] == "phone":
                    worker, worker_pid = launch_phone_worker(route, batch, raw_prefix)
                    if worker_pid:
                        worker_pids.add(worker_pid)
                run = run_host_process(route_name, route, batch, warmup, measured, raw_prefix)
                rc_all = rc_all or (0 if run["rc"] == 0 else run["rc"])
                for hc in run["host_certs"]:
                    if not placement_ok(hc):
                        host_placement_ok = False
                    compute_nodes_total += int(hc.get("compute_nodes", 0))
                    for op in cpu_ops(hc):
                        if op != "GET_ROWS":
                            cpu_get_rows_only = False
                for r in run["results"]:
                    if r["warmup"]:
                        continue
                    res = r["result"]
                    if res.get("outcome") != "completed":
                        support = False
                        err_reason = "unsupported"
                        continue
                    all_walls.append(int(res["route_wall_us"]))
                    toks = res.get("token_ids") or []
                    if toks:
                        all_tokens.append(toks[0] if isinstance(toks[0], list) else toks)
                if route["kind"] == "phone":
                    worker.wait_line("stderr", "[stagenet] exit after", 30) if False else None
                    time.sleep(0.3)
                    certs = parse_cert(worker.stderr_lines + worker.stdout_lines, "SESSIONCERT")
                    session_certs.extend(certs)
                    for c in certs:
                        worker_pids.add(str(c.get("worker_pid")))
                        worker_nonces.add(str(c.get("worker_boot_nonce")))
                        reset_flags.append(bool(c.get("reset_applied")))
                        if not placement_ok(c):
                            host_placement_ok = False
                        for op in cpu_ops(c):
                            if op != "GET_ROWS":
                                cpu_get_rows_only = False
                    (raw_prefix.with_suffix(".worker.log")).write_text(
                        "\n".join(worker.stderr_lines + worker.stdout_lines))
            except Exception as exc:  # noqa: BLE001
                support = False
                msg = str(exc)
                low = msg.lower()
                if "out of memory" in low or "oom" in low or "cuda error" in low or "alloc" in low:
                    oom = True
                    err_reason = "oom"
                else:
                    err_reason = err_reason or "unsupported"
                log(f"    rep{rep} error: {msg}")
            finally:
                if route["kind"] == "phone":
                    kill_worker(route)
                    time.sleep(0.5)
    finally:
        sampler.stop()

    gpu_after = nvml_snapshot()
    thermal_end = phone_thermal_c(route["serial"]) if route["kind"] == "phone" else None

    # oom detection from raw logs (host stderr) as a backstop
    for logf in (out_dir / "raw").glob(f"{cell_id}.rep*.host.stderr.log"):
        t = logf.read_text().lower()
        if "out of memory" in t or "failed to allocate" in t or "cuda error" in t:
            oom = True
            support = False
            err_reason = "oom"

    p50 = percentile(all_walls, 0.50)
    p95 = percentile(all_walls, 0.95)
    p99 = percentile(all_walls, 0.99)
    thr = (batch * N_GEN) / (p50 / 1e6) if p50 > 0 else 0.0
    gpu1_busy = gpu_after.get(SECOND_GPU, {}).get("util", 0) > 5 or sampler.gpu1_util_max > 5

    row = {
        "schema": "s19-batch-atlas-row-v1",
        "route": route_name,
        "device": route["device"],
        "backend": route["backend"],
        "layer_range": route["layer_range"],
        "host_layer_start": route["host_layer_start"],
        "model_hashes": {"full": FULL_MODEL_HASH,
                         "head": route.get("head_hash")},
        "binary_hashes": {"host": HOST_BIN_HASH,
                          "phone": PHONE_BIN_HASH if route["kind"] == "phone" else None},
        "context_envelope": {"context": CONTEXT, "max_prefill": MAX_PREFILL,
                             "n_gen": N_GEN, "prompt_sha256": PROMPT_SHA},
        "batch": batch,
        "support": support,
        "memory_ok": not oom,
        "oom": oom,
        "selected_gpu_peak_mib": sampler.peak,
        "correctness": {"route_token_ids": all_tokens[0] if all_tokens else [],
                        "all_route_tokens_identical": all(t == all_tokens[0] for t in all_tokens) if all_tokens else False,
                        "n_token_samples": len(all_tokens)},
        "placement": {"scheduled_placement_ok": host_placement_ok,
                      "missing_buffer": 0 if host_placement_ok else 1,
                      "cpu_get_rows_only": cpu_get_rows_only,
                      "compute_nodes": compute_nodes_total},
        "latency_us": {"p50": p50, "p95": p95, "p99": p99,
                       "route_wall_samples": all_walls},
        "throughput_tok_s": thr,
        "thermal": {"phone_start_c": thermal_start, "phone_end_c": thermal_end},
        "process_evidence": {"screen_processes": replicates,
                             "measured_exchanges": len(all_walls),
                             "worker_pids": sorted(worker_pids),
                             "worker_boot_nonces": sorted(worker_nonces),
                             "reset_applied_all": all(reset_flags) if reset_flags else None,
                             "n_session_certs": len(session_certs)},
        "gpu1": {"before": gpu_before.get(SECOND_GPU),
                 "after": gpu_after.get(SECOND_GPU),
                 "util_max_during": sampler.gpu1_util_max, "busy": gpu1_busy},
        "rc": rc_all,
        "ineligible_reason": err_reason,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True, choices=list(ROUTES))
    ap.add_argument("--batches", required=True, help="comma-separated batch sizes")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--measured", type=int, default=3)
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "results" / "atlas_rows.jsonl"

    # verify frozen host binary hash equals the executed binary hash
    import hashlib
    exec_hash = hashlib.sha256(HOST_BIN.read_bytes()).hexdigest()
    if exec_hash != HOST_BIN_HASH:
        log(f"WARNING: executed host binary hash {exec_hash} != pinned {HOST_BIN_HASH}")

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    for b in batches:
        row = measure_cell(args.route, b, args.warmup, args.measured, args.replicates, out_dir)
        row["tag"] = args.tag
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        verdict = "SUPPORT" if row["support"] else f"FAIL({row['ineligible_reason']})"
        log(f"  {args.route}.B{b}: {verdict} p50={row['latency_us']['p50']/1000:.1f}ms "
            f"thr={row['throughput_tok_s']:.1f}tok/s peakGPU={row['selected_gpu_peak_mib']}MiB "
            f"nsamp={row['process_evidence']['measured_exchanges']}")


if __name__ == "__main__":
    main()
