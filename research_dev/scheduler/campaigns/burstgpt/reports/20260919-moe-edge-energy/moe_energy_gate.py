#!/usr/bin/env python3
"""MoE host-energy gate: one llama-server arm, one or more identical requests, energy per phase.

Measures, for a MoE model whose experts live on the host (``-ot exps=CPU``), the CPU-package (RAPL)
and GPU-board (NVML via nvidia-smi) energy of prefill and decode, the server's disk reads and major
faults, and the cgroup memory events when the arm runs under a ``MemoryMax`` scope. Arms differ
only in the memory scope and page-cache state; the request is identical.

This measures the host. It does not execute or estimate any phone work.
"""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import urllib.request

RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"


def read_int(path, default=0):
    try:
        return int(pathlib.Path(path).read_text().strip())
    except (OSError, ValueError):
        return default


def proc_stat_fields(pid):
    try:
        text = pathlib.Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    rest = text[text.rindex(")") + 2:].split()
    # fields after comm: state(0) ppid(1) ... minflt(7) cminflt(8) majflt(9) cmajflt(10) utime(11) stime(12) ... rss(21)
    return {"majflt": int(rest[9]), "utime": int(rest[11]), "stime": int(rest[12]), "rss_pages": int(rest[21])}


def proc_io(pid):
    out = {}
    try:
        for line in pathlib.Path(f"/proc/{pid}/io").read_text().splitlines():
            k, v = line.split(":")
            out[k.strip()] = int(v)
    except OSError:
        pass
    return out


def proc_status_kib(pid, key):
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def cgroup_of(pid):
    try:
        line = pathlib.Path(f"/proc/{pid}/cgroup").read_text().strip()
        return "/sys/fs/cgroup" + line.split(":", 2)[2]
    except OSError:
        return None


def cgroup_stats(cg):
    out = {}
    if not cg:
        return out
    for name in ("memory.current", "memory.peak", "memory.max"):
        try:
            out[name] = pathlib.Path(cg, name).read_text().strip()
        except OSError:
            pass
    try:
        for line in pathlib.Path(cg, "memory.events").read_text().splitlines():
            k, v = line.split()
            out["events." + k] = int(v)
    except OSError:
        pass
    try:
        for line in pathlib.Path(cg, "memory.stat").read_text().splitlines():
            k, v = line.split()
            if k in ("file", "anon", "file_mapped", "pgmajfault", "pgfault", "workingset_refault_file"):
                out["stat." + k] = int(v)
    except OSError:
        pass
    return out


class Sampler(threading.Thread):
    def __init__(self, pid, period_s, cg):
        super().__init__(daemon=True)
        self.pid = pid
        self.period_s = period_s
        self.cg = cg
        self.samples = []
        self.stop_event = threading.Event()
        self.gpu_proc = None
        self.gpu_lock = threading.Lock()
        self.last_gpu = None

    def gpu_reader(self):
        try:
            self.gpu_proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", str(int(self.period_s * 1000))],
                stdout=subprocess.PIPE, text=True)
        except OSError:
            return
        for line in self.gpu_proc.stdout:
            try:
                p, u, m = [x.strip() for x in line.split(",")]
                with self.gpu_lock:
                    self.last_gpu = {"power_w": float(p), "util_pct": float(u), "mem_mib": float(m), "t": time.monotonic()}
            except ValueError:
                continue

    def run(self):
        t = threading.Thread(target=self.gpu_reader, daemon=True)
        t.start()
        while not self.stop_event.is_set():
            st = proc_stat_fields(self.pid) or {}
            with self.gpu_lock:
                gpu = dict(self.last_gpu) if self.last_gpu else None
            sample = {
                "t": time.monotonic(),
                "rapl_uj": read_int(RAPL),
                "gpu": gpu,
                "majflt": st.get("majflt"),
                "cpu_ticks": (st.get("utime", 0) + st.get("stime", 0)),
                "rss_anon_kib": proc_status_kib(self.pid, "RssAnon"),
                "rss_file_kib": proc_status_kib(self.pid, "RssFile"),
                "io_read_bytes": proc_io(self.pid).get("read_bytes"),
                "cg_current": read_int(os.path.join(self.cg, "memory.current")) if self.cg else None,
            }
            self.samples.append(sample)
            self.stop_event.wait(self.period_s)
        if self.gpu_proc:
            self.gpu_proc.terminate()

    def stop(self):
        self.stop_event.set()
        self.join(timeout=5)


def energy_in_window(samples, t0, t1, rapl_max):
    """CPU-package joules and GPU joules between t0 and t1 (linear interpolation at the edges)."""
    pts = [s for s in samples if s["t"] >= t0 - 0.5 and s["t"] <= t1 + 0.5]
    if len(pts) < 2:
        return {"cpu_j": None, "gpu_j": None, "samples": len(pts)}
    def rapl_at(t):
        for a, b in zip(pts, pts[1:]):
            if a["t"] <= t <= b["t"]:
                f = (t - a["t"]) / max(b["t"] - a["t"], 1e-9)
                d = b["rapl_uj"] - a["rapl_uj"]
                if d < 0:
                    d += rapl_max
                return a["rapl_uj"] + f * d
        return pts[-1]["rapl_uj"] if t > pts[-1]["t"] else pts[0]["rapl_uj"]
    def unwrap(a, b):
        d = b - a
        return d + rapl_max if d < 0 else d
    cpu_j = unwrap(rapl_at(t0), rapl_at(t1)) / 1e6
    gpu_j = 0.0
    gpu_n = 0
    for a, b in zip(pts, pts[1:]):
        lo, hi = max(a["t"], t0), min(b["t"], t1)
        if hi <= lo or not a["gpu"]:
            continue
        gpu_j += a["gpu"]["power_w"] * (hi - lo)
        gpu_n += 1
    return {"cpu_j": cpu_j, "gpu_j": gpu_j if gpu_n else None, "samples": len(pts)}


def http_json(url, payload=None, timeout=36000):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def drop_file_cache(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def file_cached_bytes(path):
    """Resident page-cache bytes of a file via mincore(2) over a private read-only mapping."""
    import ctypes
    import mmap
    try:
        size = os.path.getsize(path)
        if size == 0:
            return 0
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
        page = mmap.PAGESIZE
        fd = os.open(path, os.O_RDONLY)
        try:
            addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_PRIVATE, fd, 0)
        finally:
            os.close(fd)
        if addr is None or addr == ctypes.c_void_p(-1).value:
            return None
        try:
            n_pages = (size + page - 1) // page
            vec = ctypes.create_string_buffer(n_pages)
            if libc.mincore(addr, size, vec) != 0:
                return None
            return sum(1 for b in vec.raw if b & 1) * page
        finally:
            libc.munmap(addr, size)
    except (OSError, ValueError, AttributeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--lib-dir", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--memory-max", type=int, default=0, help="cgroup MemoryMax in bytes (0 = none)")
    ap.add_argument("--drop-cache", action="store_true", help="evict the model file from the page cache before launch")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--override-tensor", default="exps=CPU")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ubatch", type=int, default=512)
    ap.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    ap.add_argument("--period", type=float, default=0.1)
    ap.add_argument("--idle-s", type=float, default=15.0)
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rapl_max = read_int(RAPL_MAX, 2**32)
    record = {"arm": args.arm, "args": vars(args), "schema": "moe-energy-gate-v1", "host": os.uname().nodename}

    # Idle floor before launch (nothing of ours running).
    idle = Sampler(os.getpid(), args.period, None)
    idle.start()
    time.sleep(args.idle_s)
    idle.stop()
    t0, t1 = idle.samples[0]["t"], idle.samples[-1]["t"]
    e = energy_in_window(idle.samples, t0, t1, rapl_max)
    record["idle"] = {"seconds": t1 - t0, "cpu_w": e["cpu_j"] / (t1 - t0) if e["cpu_j"] else None,
                      "gpu_w": e["gpu_j"] / (t1 - t0) if e["gpu_j"] else None}

    if args.drop_cache:
        drop_file_cache(args.model)
    record["model_cached_bytes_before_launch"] = file_cached_bytes(args.model)

    import socket
    if args.port == 0:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            args.port = sock.getsockname()[1]
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit(f"port {args.port} already in use: a previous server is still running")
    record["port"] = args.port
    cmd = [args.server, "-m", args.model, "--port", str(args.port), "--host", "127.0.0.1",
           "-ngl", str(args.ngl), "-t", str(args.threads), "-c", str(args.ctx), "-b", str(args.batch), "-ub", str(args.ubatch),
           "--parallel", "1", "--no-warmup", "--metrics"]
    if args.override_tensor:
        cmd += ["-ot", args.override_tensor]
    cmd += args.extra
    env = dict(os.environ)
    if args.lib_dir:
        env["LD_LIBRARY_PATH"] = args.lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    if args.memory_max:
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={args.memory_max}", "-p", "MemorySwapMax=0", "--"] + cmd
    record["command"] = cmd
    log = open(out / "server.log", "w")
    t_launch = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    # Find the llama-server pid (systemd-run --scope execs in place, so proc.pid is the server).
    server_pid = proc.pid
    cg = cgroup_of(server_pid)
    sampler = Sampler(server_pid, args.period, cg)
    sampler.start()

    base = f"http://127.0.0.1:{args.port}"
    try:
        return run_arm(args, record, proc, server_pid, cg, sampler, base, out, rapl_max, t_launch)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=60)
        sampler.stop()
        record["server_exit"] = proc.returncode
        (out / "SAMPLES.json").write_text(json.dumps(sampler.samples))
        (out / "RESULT.json").write_text(json.dumps(record, indent=2))


def run_arm(args, record, proc, server_pid, cg, sampler, base, out, rapl_max, t_launch):
    ready = False
    while time.monotonic() - t_launch < 3600:
        if proc.poll() is not None:
            break
        try:
            h = http_json(base + "/health", timeout=5)
            if h.get("status") == "ok":
                ready = True
                break
        except Exception:
            time.sleep(0.5)
    record["load_s"] = time.monotonic() - t_launch
    time.sleep(1.0)
    if proc.poll() is not None:
        ready = False
    record["ready"] = ready
    if ready:
        # systemd-run moves the process into its transient scope after exec; resolve the cgroup now.
        cg = cgroup_of(server_pid)
        sampler.cg = cg
        record["cgroup"] = cg
        cmdline = pathlib.Path(f"/proc/{server_pid}/cmdline").read_bytes().split(b"\0")
        record["server_cmdline"] = [c.decode(errors="replace") for c in cmdline if c]
        if args.model.encode() not in b" ".join(cmdline):
            raise SystemExit("the process on the port is not our server")
    if not ready:
        print(json.dumps({"arm": args.arm, "ready": False, "server_exit": proc.poll()}), flush=True)
        return 2

    text = pathlib.Path(args.prompt_file).read_text(errors="replace")[: args.prompt_tokens * 8]
    toks = http_json(base + "/tokenize", {"content": text})["tokens"]
    toks = toks[: args.prompt_tokens]
    record["prompt_tokens"] = len(toks)

    requests = []
    for i in range(args.repeat):
        before = {"stat": proc_stat_fields(server_pid), "io": proc_io(server_pid), "cg": cgroup_stats(cg),
                  "model_cached_bytes": file_cached_bytes(args.model)}
        t_req = time.monotonic()
        resp = http_json(base + "/completion", {"prompt": toks, "n_predict": args.n_predict, "temperature": 0.0,
                                                "cache_prompt": False, "n_probs": 0, "stream": False})
        t_end = time.monotonic()
        after = {"stat": proc_stat_fields(server_pid), "io": proc_io(server_pid), "cg": cgroup_stats(cg),
                 "model_cached_bytes": file_cached_bytes(args.model)}
        timings = resp.get("timings", {})
        prompt_s = timings.get("prompt_ms", 0) / 1000.0
        # Server timings start after request parsing; anchor the prefill window at the request start.
        t_pre_end = t_req + prompt_s
        pre = energy_in_window(sampler.samples, t_req, t_pre_end, rapl_max)
        dec = energy_in_window(sampler.samples, t_pre_end, t_end, rapl_max)
        def sample_at(t, key):
            best = None
            for smp in sampler.samples:
                if smp.get(key) is None:
                    continue
                if best is None or abs(smp["t"] - t) < abs(best["t"] - t):
                    best = smp
            return best.get(key) if best else None
        decode_reads = (sample_at(t_end, "io_read_bytes") or 0) - (sample_at(t_pre_end, "io_read_bytes") or 0)
        decode_majflt = (sample_at(t_end, "majflt") or 0) - (sample_at(t_pre_end, "majflt") or 0)
        req = {
            "index": i, "request_s": t_end - t_req, "timings": timings,
            "t_request": t_req, "t_prefill_end": t_pre_end, "t_end": t_end,
            "decode_read_bytes": decode_reads, "decode_majflt": decode_majflt,
            "content": resp.get("content", ""), "tokens_predicted": resp.get("tokens_predicted"),
            "prefill": {"seconds": prompt_s, **pre},
            "decode": {"seconds": t_end - t_pre_end, "tokens": timings.get("predicted_n"), **dec},
            "majflt_delta": (after["stat"] or {}).get("majflt", 0) - (before["stat"] or {}).get("majflt", 0),
            "read_bytes_delta": after["io"].get("read_bytes", 0) - before["io"].get("read_bytes", 0),
            "before": before, "after": after,
        }
        n = req["decode"]["tokens"] or 1
        if dec["cpu_j"] is not None:
            req["decode"]["cpu_j_per_token"] = dec["cpu_j"] / n
            req["decode"]["gpu_j_per_token"] = (dec["gpu_j"] or 0) / n
            req["decode"]["host_j_per_token"] = (dec["cpu_j"] + (dec["gpu_j"] or 0)) / n
            req["decode"]["ms_per_token"] = timings.get("predicted_per_token_ms")
        requests.append(req)
        print(json.dumps({"arm": args.arm, "request": i, "prefill_s": round(prompt_s, 2),
                          "decode_ms_per_token": timings.get("predicted_per_token_ms"),
                          "host_j_per_token": req["decode"].get("host_j_per_token"),
                          "read_bytes_delta_gib": round(req["read_bytes_delta"] / 2**30, 3),
                          "decode_read_gib": round(decode_reads / 2**30, 3), "decode_majflt": decode_majflt,
                          "majflt_delta": req["majflt_delta"]}), flush=True)

    record["requests"] = requests
    record["cgroup_final"] = cgroup_stats(cg)
    record["server_rss_anon_kib"] = proc_status_kib(server_pid, "RssAnon")
    record["server_rss_file_kib"] = proc_status_kib(server_pid, "RssFile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
