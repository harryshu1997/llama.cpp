#!/usr/bin/env python3
"""Stronger-server FFN column-split benchmark (Stage S1 of ../PLAN.md).

Subcommands
    run    start llama-server (build-cuda-s43, GPU 0) once per arm, drive a fixed workload, write
           RESULT-<arm>.json + samples-<arm>.json + SUMMARY.md into --out
    probe  talk to every helper worker directly (HELLO geometry + EXECUTE round trips at rows 1/2/4)
    plan   print the exact server argv/env and control payloads per arm, launch nothing

Arms (benchlib.parse_arm): cpu | gpu | cpu-helpers | phone | split-25 | split-50 | split-75 | split-NN
    cpu          no FFN runtime: the server CPU runs every CPU-resident layer (baseline, runs without phones)
    phone        phones own 100 % of the FFN columns of layers 0-23 (desktop mode)
    split-NN     phones own NN % of the columns (trailing suffix), the host CPU the rest, concurrently
    cpu-helpers  FFN runtime configured but never activated (control for the runtime's own overhead)
    gpu          every layer on the GPU (reference only: the model fits one A6000)

Placement: --n-gpu-layers 16 keeps layers 0-23 (and the output head) CPU-resident, as on the desktop.
This is a MECHANISM test; on this server the 29.5 GB model fits in one A6000's 48 GB.

Phone arms use the decode-boundary runtime control (as the desktop scheduler): the server starts with
deferred helper connections, every request carries X-Scheduler-Request-ID and a pinned id_slot, and
after its first streamed token the harness posts the policy to POST /v1/chat/completions/control
(action ffn_split, or ffn_split_cohort for concurrent requests). Prefill always runs on the host.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchlib as bl  # noqa: E402

PROMPTS = [
    "The following is an engineering note about memory bandwidth in large language model inference. "
    "During autoregressive decoding every generated token must read all model weights once, so the "
    "token rate is bounded by how fast the weights stream from memory rather than by arithmetic. "
    "A 14-billion-parameter model stored in 16-bit floats occupies about 28 GB, and a processor that "
    "streams 70 GB/s therefore produces at most about 2.5 tokens per second. Offloading part of the "
    "layers to a graphics card helps because its memory is ten times faster, but the layers that stay "
    "on the host still dominate. Explain, step by step, how splitting the feed-forward columns of each "
    "host layer between the host and a second device changes the time per token, and what role the "
    "round-trip latency of the link plays.",
    "Write a detailed tutorial on configuring a home WiFi 7 network for low-latency device-to-device "
    "traffic. Cover channel selection on the 6 GHz band, multi-link operation, the difference between "
    "access-point mode and router mode, why client isolation must be disabled, how to measure round-trip "
    "time and jitter with small request-response messages, and how power-saving modes on phones add tail "
    "latency. Include concrete commands for Linux and Android where possible, and end with a checklist.",
    "Summarize the history of the transistor from the point-contact device of 1947 to modern gate-all-"
    "around nanosheets. Discuss the planar MOSFET, the role of Dennard scaling and why it ended, the "
    "introduction of high-k metal gates, the move to FinFETs at the 22 nm node, and the reasons the "
    "industry is now adopting nanosheet transistors and backside power delivery. For each step explain "
    "the physical problem that forced the change and the trade-offs the new structure introduced.",
    "You are reviewing a pull request that adds a request scheduler to an inference server. The scheduler "
    "batches decode steps of concurrent requests, offloads some matrix products to remote helpers, and "
    "must never block the main loop on a slow helper. List the correctness risks you would check, the "
    "metrics you would ask the author to report, the failure modes of a helper that disconnects in the "
    "middle of a token, and the tests you would require before merging. Be specific and practical.",
    "Describe how a city could plan a district heating network that uses waste heat from a data center. "
    "Explain how the heat is captured from liquid-cooled servers, the temperature levels involved, the "
    "role of heat pumps, how pipes are sized and insulated, how demand varies over a day and a year, what "
    "storage options exist, and how the operator and the data center would share costs and risks. Close "
    "with the three decisions that matter most for the project's economics.",
]


# ---------------------------------------------------------------------------------------------
# power / counters

class Nvml:
    """Minimal NVML via ctypes: board power (mW) and the total-energy counter (mJ, Volta+)."""

    def __init__(self, indices):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit failed")
        self.handles = {}
        for index in indices:
            handle = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                raise OSError("no NVML device %d" % index)
            self.handles[index] = handle

    def read(self, index):
        power, energy = ctypes.c_uint(), ctypes.c_ulonglong()
        handle = self.handles[index]
        ok_power = self.lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power)) == 0
        ok_energy = self.lib.nvmlDeviceGetTotalEnergyConsumption(handle, ctypes.byref(energy)) == 0
        return (power.value / 1000.0 if ok_power else None, energy.value / 1000.0 if ok_energy else None)

    def memory_used_mib(self, index):
        class Memory(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
        memory = Memory()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handles[index], ctypes.byref(memory)) != 0:
            return None
        return memory.used / 2**20, memory.free / 2**20


RAPL_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"


def rapl_status():
    try:
        with open(RAPL_PATH) as handle:
            int(handle.read())
        wrap = None
        try:
            with open(os.path.join(os.path.dirname(RAPL_PATH), "max_energy_range_uj")) as handle:
                wrap = int(handle.read())
        except OSError:
            pass
        return {"available": True, "path": RAPL_PATH, "wrap_uj": wrap}
    except PermissionError:
        return {"available": False, "path": RAPL_PATH,
                "reason": "permission denied (energy_uj is root-only on FCHLLX01); CPU package energy recorded as null"}
    except OSError as error:
        return {"available": False, "path": RAPL_PATH, "reason": str(error)}


def read_proc_stat():
    with open("/proc/stat") as handle:
        fields = handle.readline().split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


class Sampler(threading.Thread):
    """5 Hz: GPU board power + NVML energy counter, RAPL package energy (if readable), CPU busy."""

    def __init__(self, gpu_indices, period_s=0.2):
        super().__init__(daemon=True)
        self.period_s = period_s
        self.gpu_indices = list(gpu_indices)
        self.stop_event = threading.Event()
        self.rows = []
        self.rapl = rapl_status()
        try:
            self.nvml = Nvml(self.gpu_indices)
            self.nvml_error = None
        except OSError as error:
            self.nvml, self.nvml_error = None, str(error)

    def run(self):
        rapl_prev, rapl_total = None, 0
        while not self.stop_event.is_set():
            t = time.monotonic()
            row = {"t": t}
            if self.nvml:
                for index in self.gpu_indices:
                    power, energy = self.nvml.read(index)
                    row["gpu%d_w" % index], row["gpu%d_j" % index] = power, energy
            if self.rapl["available"]:
                try:
                    with open(RAPL_PATH) as handle:
                        value = int(handle.read())
                    if rapl_prev is not None:
                        rapl_total += bl.counter_delta(rapl_prev, value, self.rapl.get("wrap_uj"))
                    rapl_prev = value
                    row["cpu_pkg_j"] = rapl_total / 1e6
                except (OSError, ValueError):
                    row["cpu_pkg_j"] = None
            total, idle = read_proc_stat()
            row["cpu_total_jiffies"], row["cpu_idle_jiffies"] = total, idle
            self.rows.append(row)
            self.stop_event.wait(max(0.0, self.period_s - (time.monotonic() - t)))

    def stop(self):
        self.stop_event.set()
        self.join(timeout=5)

    def series(self, key):
        return [(row["t"], row[key]) for row in self.rows if row.get(key) is not None]

    def window(self, t0, t1):
        out = {"t0": t0, "t1": t1, "seconds": t1 - t0}
        for index in self.gpu_indices:
            out["gpu%d_j" % index] = bl.energy_between(self.series("gpu%d_j" % index), t0, t1)
            out["gpu%d_mean_w" % index] = bl.mean_between(self.series("gpu%d_w" % index), t0, t1)
        out["cpu_pkg_j"] = bl.energy_between(self.series("cpu_pkg_j"), t0, t1) if self.rapl["available"] else None
        inside = [row for row in self.rows if t0 <= row["t"] <= t1]
        if len(inside) >= 2:
            dt = inside[-1]["cpu_total_jiffies"] - inside[0]["cpu_total_jiffies"]
            di = inside[-1]["cpu_idle_jiffies"] - inside[0]["cpu_idle_jiffies"]
            out["host_cpu_busy_frac"] = None if dt <= 0 else 1.0 - di / dt
        else:
            out["host_cpu_busy_frac"] = None
        return out


# ---------------------------------------------------------------------------------------------
# llama-server process

class Server:
    def __init__(self, argv, env, log_path, ffn_lines_path):
        self.argv, self.env = argv, env
        self.log_path, self.ffn_lines_path = log_path, ffn_lines_path
        self.proc = None
        self.ffn = []          # (t_monotonic, parsed row)
        self.tail = []
        self.lock = threading.Lock()

    def start(self):
        self.log = open(self.log_path, "w")
        self.ffn_log = open(self.ffn_lines_path, "w")
        self.proc = subprocess.Popen(self.argv, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, bufsize=1, text=True, errors="replace",
                                     start_new_session=True)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.proc.stdout:
            t = time.monotonic()
            self.log.write(line)
            row = bl.parse_ffn_line(line)
            with self.lock:
                self.tail = (self.tail + [line.rstrip()])[-60:]
                if row is not None:
                    self.ffn.append((t, row))
                    self.ffn_log.write("%.6f\t%s" % (t, line if line.endswith("\n") else line + "\n"))
        self.log.flush()

    def wait_ready(self, url_host, port, timeout_s=900):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited with %s:\n%s" % (self.proc.returncode, "\n".join(self.tail)))
            try:
                conn = http.client.HTTPConnection(url_host, port, timeout=2)
                conn.request("GET", "/health")
                response = conn.getresponse()
                response.read()
                conn.close()
                if response.status == 200:
                    return
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError("llama-server not ready after %d s" % timeout_s)

    def stop(self, timeout_s=120):
        if self.proc is None or self.proc.poll() is not None:
            return self.proc.returncode if self.proc else None
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self.reader.join(timeout=10)
        self.log.close()
        self.ffn_log.close()
        return self.proc.returncode


# ---------------------------------------------------------------------------------------------
# tap proxy: per-call timing for TCP helpers (the server prints S41SERVERFFNUSB timing lines only
# for FunctionFS; over TCP it prints S41SERVERFFNCALL proofs without times)

def _recv_exact(sock, size):
    buffer = bytearray(size)
    view, got = memoryview(buffer), 0
    while got < size:
        count = sock.recv_into(view[got:], size - got)
        if count == 0:
            raise ConnectionError("peer closed")
        got += count
    return bytes(buffer)


class TapProxy(threading.Thread):
    """127.0.0.1:listen_port -> target host:port, protocol-aware; logs one row per EXECUTE call."""

    def __init__(self, label, listen_port, target_host, target_port):
        super().__init__(daemon=True)
        self.label, self.listen_port = label, listen_port
        self.target = (target_host, target_port)
        self.calls, self.errors = [], []
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", listen_port))
        self.listener.listen(4)
        self.closing = False

    def run(self):
        while not self.closing:
            try:
                client, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(client,), daemon=True).start()

    def _serve(self, client):
        upstream = None
        try:
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            upstream = socket.create_connection(self.target, timeout=30)
            upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            upstream.settimeout(None)
            upstream.sendall(_recv_exact(client, bl.HELLO_REQUEST.size))
            client.sendall(_recv_exact(upstream, bl.HELLO_RESPONSE.size))
            while True:
                header = _recv_exact(client, bl.EXECUTE_REQUEST.size)
                t_in = time.monotonic_ns()
                request = bl.decode_execute_request(header)
                payload = _recv_exact(client, request["payload_bytes"])
                upstream.sendall(header + payload)
                t_sent = time.monotonic_ns()
                response_header = _recv_exact(upstream, bl.EXECUTE_RESPONSE.size)
                response = bl.decode_execute_response(response_header)
                response_payload = _recv_exact(upstream, response["payload_bytes"])
                t_back = time.monotonic_ns()
                client.sendall(response_header + response_payload)
                rpc_ms = (t_back - t_sent) / 1e6
                compute_ms = response["compute_us"] / 1000.0
                self.calls.append({"label": self.label, "request_id": request["request_id"],
                                   "layer": request["layer"], "tokens": request["tokens"],
                                   "columns": request["columns"], "t_in_ns": t_in, "t_sent_ns": t_sent,
                                   "t_back_ns": t_back, "rpc_ms": rpc_ms, "compute_ms": compute_ms,
                                   "transport_ms": rpc_ms - compute_ms, "status": response["status"]})
        except ConnectionError:
            pass  # a side closed the session (server shutdown or worker exit): normal end
        except OSError as error:
            if not self.closing:
                self.errors.append(str(error))
        finally:
            for sock in (client, upstream):
                try:
                    if sock:
                        sock.close()
                except OSError:
                    pass

    def close(self):
        self.closing = True
        try:
            self.listener.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------
# helper probe (HELLO + EXECUTE)

def hello_probe(helper, server, timeout_s=10.0):
    sock = socket.create_connection((helper["host"], helper["port"]), timeout=timeout_s)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.sendall(bl.encode_hello(helper["layer_mask"], server["n_embd"], server["n_ff"], server["max_tokens"],
                                 server["artifact_sha256"], True, server["activation"] == "swiglu"))
    response = bl.decode_hello_response(_recv_exact(sock, bl.HELLO_RESPONSE.size))
    return sock, response


def probe_helper(helper, server, calls=200, rows=(1, 2, 4), gap_ms=8.0, columns=None):
    out = {"label": helper["label"], "host": helper["host"], "port": helper["port"], "layers": helper["layers"]}
    try:
        sock, hello = hello_probe(helper, server)
    except OSError as error:
        out["error"] = "connect/HELLO failed: %s" % error
        return out
    out["hello"] = hello
    problems = []
    if not hello["ok"]:
        problems.append("worker rejected HELLO (status %d): artifact/layers/columns/max_tokens/flags differ"
                        % hello["status"])
    if hello["n_ff"] != server["n_ff"] or hello["offset"] != 0:
        problems.append("geometry n_ff=%d offset=%d (server needs n_ff=%d offset=0)"
                        % (hello["n_ff"], hello["offset"], server["n_ff"]))
    if server["column_quantum"] % max(1, hello["column_quantum"]):
        problems.append("worker quantum %d does not divide the harness quantum %d"
                        % (hello["column_quantum"], server["column_quantum"]))
    out["problems"] = problems
    if problems or calls <= 0:
        sock.close()
        return out
    columns = columns or server["n_ff"]
    layers = bl.mask_layers(helper["layer_mask"])
    payloads = {r: bl.probe_payload(server["n_embd"], r, seed=r) for r in rows}
    hashes = {r: bl.fnv1a32(payloads[r]) for r in rows}
    results = {}
    request_id = 1
    sock.settimeout(60)
    for r in rows:
        if r > server["max_tokens"]:
            continue
        samples = []
        for index in range(calls):
            layer = layers[index % len(layers)]
            header = bl.encode_execute(request_id, layer, r, server["n_embd"], columns, payloads[r], True, hashes[r])
            t0 = time.perf_counter_ns()
            sock.sendall(header + payloads[r])
            response = bl.decode_execute_response(_recv_exact(sock, bl.EXECUTE_RESPONSE.size))
            _recv_exact(sock, response["payload_bytes"])
            rpc_ms = (time.perf_counter_ns() - t0) / 1e6
            if not response["ok"] or response["request_id"] != request_id:
                problems.append("EXECUTE failed at call %d rows %d" % (index, r))
                break
            samples.append({"rpc_ms": rpc_ms, "compute_ms": response["compute_us"] / 1000.0,
                            "transport_ms": rpc_ms - response["compute_us"] / 1000.0, "layer": layer})
            request_id += 1
            if gap_ms > 0:
                time.sleep(gap_ms / 1000.0)
        results[str(r)] = bl.summarize_calls(samples)
    sock.close()
    out["rows"] = results
    return out


# ---------------------------------------------------------------------------------------------
# requests + control

class Request(threading.Thread):
    def __init__(self, host, port, rid, slot, prompt, n_predict, first_token_event, start_barrier):
        super().__init__(daemon=True)
        self.host, self.port, self.rid, self.slot = host, port, rid, slot
        self.prompt, self.n_predict = prompt, n_predict
        self.first_token_event, self.start_barrier = first_token_event, start_barrier
        self.token_times, self.token_ids, self.final, self.error = [], [], None, None
        self.t_send = None

    def run(self):
        body = json.dumps({"prompt": self.prompt, "n_predict": self.n_predict, "temperature": 0.0, "top_k": 1,
                           "seed": 1234, "stream": True, "return_tokens": True, "ignore_eos": True,
                           "cache_prompt": False, "id_slot": self.slot})
        try:
            self.start_barrier.wait(timeout=60)
            conn = http.client.HTTPConnection(self.host, self.port, timeout=1800)
            self.t_send = time.monotonic()
            conn.request("POST", "/completion", body, {"Content-Type": "application/json",
                                                        "X-Scheduler-Request-ID": self.rid})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError("HTTP %d: %s" % (response.status, response.read()[:500]))
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                chunk = json.loads(line[6:])
                t = time.monotonic()
                tokens = chunk.get("tokens") or []
                for token in tokens:
                    self.token_ids.append(token)
                    self.token_times.append(t)
                if tokens and not self.first_token_event.is_set():
                    self.first_token_event.set()
                if chunk.get("stop"):
                    chunk.pop("generation_settings", None)
                    self.final = chunk
                    break
            conn.close()
        except Exception as error:  # noqa: BLE001 - recorded in the result
            self.error = repr(error)
            self.first_token_event.set()


def post_json(host, port, path, payload, timeout=120):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
    response = conn.getresponse()
    text = response.read().decode("utf-8", "replace")
    conn.close()
    try:
        data = json.loads(text)
    except ValueError:
        data = {"raw": text}
    return response.status, data


def control_payloads(requests, layer_mask, columns, arm_name, generation=1):
    digest = bl.policy_hash(layer_mask, columns, arm_name)
    common = {"policy_hash": digest, "plan_generation": generation, "layer_mask": layer_mask,
              "columns": columns, "enabled": layer_mask != 0 and columns != 0}
    if len(requests) == 1:
        return [dict(common, action="ffn_split", request_id=requests[0][0], slot_id=requests[0][1])]
    return [dict(common, action="ffn_split_cohort",
                 members=[{"request_id": rid, "slot_id": slot} for rid, slot in requests])]


def apply_control(host, port, requests, layer_mask, columns, arm_name, attempts=40):
    """Post the policy once every request of the wave decodes; per-slot fallback if the cohort fails."""
    log = []
    for payload in control_payloads(requests, layer_mask, columns, arm_name):
        for attempt in range(attempts):
            t = time.monotonic()
            status, data = post_json(host, port, "/v1/chat/completions/control", payload)
            ok = status == 200 and data.get("success", False)
            rids = [payload["request_id"]] if "request_id" in payload else [m["request_id"] for m in payload["members"]]
            log.append({"t": t, "action": payload["action"], "request_ids": rids, "status": status,
                        "attempt": attempt, "response": data})
            if ok:
                return True, log
            message = json.dumps(data)
            if "decode state" in message or "active slots differ" in message or "no live slot" in message:
                time.sleep(0.05)
                continue
            break
    if len(requests) > 1:
        ok_all = True
        for rid, slot in requests:
            ok, sub = apply_control(host, port, [(rid, slot)], layer_mask, columns, arm_name, attempts)
            log.extend(sub)
            ok_all = ok_all and ok
        return ok_all, log
    return False, log


def run_wave(server_cfg, arm, layer_mask, prompts, wave_index, concurrency, n_predict, run_tag):
    host, port = server_cfg["host"], server_cfg["port"]
    barrier = threading.Barrier(concurrency)
    requests = []
    for slot in range(concurrency):
        prompt_index = (wave_index * concurrency + slot) % len(prompts)
        rid = "%s-%s-c%d-w%d-s%d" % (run_tag, arm["name"], concurrency, wave_index, slot)
        request = Request(host, port, rid, slot, prompts[prompt_index], n_predict, threading.Event(), barrier)
        request.prompt_index, request.wave_index = prompt_index, wave_index
        requests.append(request)
    t_start = time.monotonic()
    for request in requests:
        request.start()
    control = {"applied": None, "log": []}
    if arm["helpers"] and arm["columns"] > 0:
        for request in requests:
            request.first_token_event.wait(timeout=900)
        live = [(request.rid, request.slot) for request in requests if request.error is None]
        if live:
            ok, log = apply_control(host, port, live, layer_mask, arm["columns"], arm["name"])
            control = {"applied": ok, "log": log}
    for request in requests:
        request.join(timeout=3600)
    t_end = time.monotonic()
    return requests, control, t_start, t_end


def applied_index(control_log, rid):
    """applied_token_index of a request from the successful control responses (cohort or single)."""
    for entry in control_log:
        data = entry.get("response", {})
        if not data.get("success") or rid not in entry.get("request_ids", []):
            continue
        for member in data.get("cohort_members", []) or []:
            if member.get("request_id") == rid:
                return member.get("applied_token_index")
        if entry["action"] == "ffn_split":
            return data.get("applied_token_index")
    return None


def summarize_wave(requests, control, t_start, t_end, sampler, skip, gpu_index):
    rows = []
    for request in requests:
        applied = applied_index(control["log"], request.rid)
        steady_skip = max(skip, (applied or 0) + 2)
        timings = (request.final or {}).get("timings", {})
        rows.append({
            "request_id": request.rid, "slot": request.slot, "prompt_index": request.prompt_index,
            "wave": request.wave_index,
            "error": request.error, "n_tokens": len(request.token_ids), "token_ids": request.token_ids,
            "ttft_ms": (request.token_times[0] - request.t_send) * 1000.0 if request.token_times and request.t_send else None,
            "applied_token_index": applied, "steady_skip": steady_skip,
            "steady_period_ms": bl.steady_period_ms(request.token_times, steady_skip),
            "server_timings": timings,
            "token_times_rel_s": [round(t - t_start, 6) for t in request.token_times],
        })
    good = [request for request in requests if request.error is None and len(request.token_times) > skip + 2]
    window = None
    if good:
        w0 = max(request.token_times[max(skip, 0)] for request in good)
        w1 = min(request.token_times[-1] for request in good)
        if w1 > w0:
            tokens = sum(sum(1 for t in request.token_times if w0 < t <= w1) for request in good)
            energy = sampler.window(w0, w1)
            window = dict(energy, tokens=tokens)
    full = dict(sampler.window(t_start, t_end), tokens=sum(len(request.token_ids) for request in requests))
    periods = sorted(row["steady_period_ms"] for row in rows if row["steady_period_ms"])
    return {"t_start": t_start, "t_end": t_end, "wall_s": t_end - t_start, "requests": rows,
            "control": control, "decode_window": window, "full_window": full,
            "step_period_ms_median": periods[len(periods) // 2] if periods else None}


def level_metrics(waves, concurrency, gpu_index):
    periods = [wave["step_period_ms_median"] for wave in waves if wave["step_period_ms_median"]]
    period = sorted(periods)[len(periods) // 2] if periods else None
    gkey = "gpu%d_j" % gpu_index
    dec_e = [wave["decode_window"][gkey] for wave in waves if wave["decode_window"] and wave["decode_window"].get(gkey) is not None]
    dec_t = [wave["decode_window"]["tokens"] for wave in waves if wave["decode_window"] and wave["decode_window"].get(gkey) is not None]
    cpu_e = [wave["decode_window"]["cpu_pkg_j"] for wave in waves if wave["decode_window"] and wave["decode_window"].get("cpu_pkg_j") is not None]
    full_e = [wave["full_window"][gkey] for wave in waves if wave["full_window"].get(gkey) is not None]
    full_t = sum(wave["full_window"]["tokens"] for wave in waves)
    wall = sum(wave["wall_s"] for wave in waves)
    watts = [wave["decode_window"]["gpu%d_mean_w" % gpu_index] for wave in waves
             if wave["decode_window"] and wave["decode_window"].get("gpu%d_mean_w" % gpu_index) is not None]
    busy = [wave["decode_window"]["host_cpu_busy_frac"] for wave in waves
            if wave["decode_window"] and wave["decode_window"].get("host_cpu_busy_frac") is not None]
    return {
        "concurrency": concurrency, "waves": len(waves),
        "step_period_ms": period, "ms_per_token": period / concurrency if period else None,
        "decode_tok_s": concurrency * 1000.0 / period if period else None,
        "e2e_tok_s": full_t / wall if wall > 0 else None, "generated_tokens": full_t, "wall_s": wall,
        "gpu_decode_j_per_token": sum(dec_e) / sum(dec_t) if dec_t and sum(dec_t) else None,
        "gpu_decode_mean_w": sum(watts) / len(watts) if watts else None,
        "cpu_pkg_decode_j_per_token": sum(cpu_e) / sum(dec_t) if cpu_e and dec_t and sum(dec_t) else None,
        "gpu_full_j_per_token": sum(full_e) / full_t if full_e and full_t else None,
        "host_cpu_busy_frac": sum(busy) / len(busy) if busy else None,
        "errors": [row["error"] for wave in waves for row in wave["requests"] if row["error"]],
        "control_ok": all(wave["control"]["applied"] in (True, None) for wave in waves),
    }


def ffn_digest(rows):
    """Aggregate parsed FFN stderr rows of one server run."""
    out = {"helpers": [], "ready": None, "caps": None, "calls": 0, "calls_by_layer": {}, "rows_hist": {},
           "columns_hist": {}, "summaries": [], "shapes": [], "errors": [], "resets": [], "controls": 0,
           "usb_calls": []}
    for _t, row in rows:
        kind = row["kind"]
        if kind == "helper":
            out["helpers"].append(row)
        elif kind == "ready":
            out["ready"] = row
        elif kind == "caps":
            out["caps"] = row
        elif kind == "call":
            out["calls"] += 1
            key = str(row.get("layer"))
            out["calls_by_layer"][key] = out["calls_by_layer"].get(key, 0) + 1
            out["rows_hist"][str(row.get("tokens"))] = out["rows_hist"].get(str(row.get("tokens")), 0) + 1
            out["columns_hist"][str(row.get("columns"))] = out["columns_hist"].get(str(row.get("columns")), 0) + 1
        elif kind == "summary":
            out["summaries"].append(row)
        elif kind == "shape":
            out["shapes"].append(row)
        elif kind == "error":
            out["errors"].append(row.get("text"))
        elif kind == "reset":
            out["resets"].append(row)
        elif kind == "control":
            out["controls"] += 1
        elif kind == "usb_call":
            out["usb_calls"].append(row)
    out["usb_call_summary"] = bl.summarize_calls(out.pop("usb_calls")) if out.get("usb_calls") else None
    return out


# ---------------------------------------------------------------------------------------------
# preflight

def library_has_ffn_split(binary):
    lib = Path(binary).resolve().parent / "libllama-server-impl.so"
    candidates = [lib, Path(binary).resolve()]
    for path in candidates:
        if path.is_file():
            data = path.read_bytes()
            if b"S41SERVERFFNCAPS" in data:
                return True, str(path)
    return False, str(lib)


def host_contention():
    try:
        load = open("/proc/loadavg").read().split()[:3]
    except OSError:
        load = None
    try:
        top = subprocess.run(["ps", "-eo", "pid,user,pcpu,comm", "--sort=-pcpu"], capture_output=True, text=True,
                             timeout=10).stdout.splitlines()[:8]
    except (OSError, subprocess.SubprocessError):
        top = []
    return {"loadavg": load, "top": top}


def port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) != 0


def resolve_binary(path):
    return path if os.path.isabs(path) else os.path.join(bl.repo_root(), path)


def sha256_file(path, chunk=64 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return "sha256:" + digest.hexdigest()


# ---------------------------------------------------------------------------------------------
# commands

def cmd_plan(config, args):
    for name in args.arms.split(","):
        arm = bl.parse_arm(name, config["server"]["n_ff"], config["server"]["column_quantum"])
        helpers = config["helpers"]
        if args.tap and arm["helpers"]:
            helpers = bl.helpers_for_tap(helpers, args.tap_base_port)
        env = bl.server_ffn_environment(config, arm, helpers)
        print("== arm %s (%s, phone columns %d = %.0f %%)" % (arm["name"], arm["kind"], arm["columns"],
                                                             100 * arm["share"]))
        print("   argv:", " ".join(bl.server_argv(config, arm, resolve_binary(config["server"]["binary"]))))
        print("   env: CUDA_VISIBLE_DEVICES=%s%s" % (config["server"]["cuda_visible_devices"],
                                                     "" if config["server"].get("cuda_graphs") else " GGML_CUDA_DISABLE_GRAPHS=1"))
        for key, value in sorted(env.items()):
            print("        %s=%s" % (key, value))
        if arm["helpers"] and arm["columns"]:
            for concurrency in args.concurrency:
                payload = control_payloads([("<rid-%d>" % i, i) for i in range(concurrency)],
                                           config["phone_layer_mask"], arm["columns"], arm["name"])
                print("   control c=%d: POST /v1/chat/completions/control %s" % (concurrency, json.dumps(payload[0])))
    return 0


def cmd_probe(config, args):
    server = config["server"]
    results = []
    for helper in config["helpers"]:
        if args.host_override:
            helper = dict(helper, host=args.host_override)
        result = probe_helper(helper, server, calls=args.calls, gap_ms=args.gap_ms)
        results.append(result)
        rows = result.get("rows", {})
        status = result.get("error") or ("; ".join(result.get("problems", [])) or "ok")
        print("%-10s %s:%d layers %-6s %s" % (helper["label"], helper["host"], helper["port"], helper["layers"], status))
        for r, summary in rows.items():
            print("    rows %s: rpc p50 %.2f p99 %.2f ms | compute p50 %.2f ms | transport p50 %.2f p99 %.2f ms"
                  % (r, summary["rpc_ms"]["p50"], summary["rpc_ms"]["p99"], summary["compute_ms"]["p50"],
                     summary["transport_ms"]["p50"], summary["transport_ms"]["p99"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({"schema": "wifi-server-helper-probe-v1", "results": results}, handle, indent=1)
    return 0 if all(not r.get("error") and not r.get("problems") for r in results) else 1


def run_arm(config, arm, args, out_dir, run_tag):
    server = dict(config["server"])
    helpers = config["helpers"]
    if args.host_override:
        helpers = [dict(helper, host=args.host_override) for helper in helpers]
    taps, tapped = [], None
    if args.tap and arm["helpers"]:
        tapped = bl.helpers_for_tap(helpers, args.tap_base_port)
        helpers = tapped
    binary = resolve_binary(server["binary"])
    argv = bl.server_argv(config, arm, binary)
    env = bl.server_process_environment(config, arm, helpers)
    if not port_free(server["host"], server["port"]):
        raise RuntimeError("port %d is busy" % server["port"])
    result = {"schema": "wifi-server-arm-result-v1", "arm": arm, "run_tag": run_tag,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "server_argv": argv, "ffn_environment": bl.server_ffn_environment(config, arm, helpers),
              "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"), "helpers": helpers if arm["helpers"] else [],
              "model": {"path": server["model"], "artifact_sha256": server["artifact_sha256"],
                        "size": os.path.getsize(server["model"])},
              "host_before": host_contention(), "workload": {
                  "prompts": args.prompts, "n_predict": args.n_predict, "concurrency": args.concurrency,
                  "skip_tokens": args.skip_tokens}, "tap": bool(tapped)}
    if arm["helpers"] and arm["columns"] > 0 and not args.no_preflight_probe:  # cpu-helpers never connects
        probes = []
        for helper in (config["helpers"] if not args.host_override else
                       [dict(h, host=args.host_override) for h in config["helpers"]]):
            probe = probe_helper(helper, server, calls=0)
            probes.append(probe)
            if probe.get("error") or probe.get("problems"):
                raise RuntimeError("helper %s preflight failed: %s" % (helper["label"],
                                                                      probe.get("error") or probe.get("problems")))
        result["helper_preflight"] = probes
    for row in tapped or []:
        taps.append(TapProxy(row["label"], row["port"], row["target_host"], row["target_port"]))
        taps[-1].start()
    sampler = Sampler([args.gpu_index])
    result["power_sources"] = {"nvml": sampler.nvml_error or "ok", "rapl": sampler.rapl}
    srv = Server(argv, env, str(out_dir / ("server-%s.log" % arm["name"])),
                 str(out_dir / ("server-%s.ffn.tsv" % arm["name"])))
    t_launch = time.monotonic()
    srv.start()
    sampler.start()
    levels = []
    try:
        srv.wait_ready(server["host"], server["port"])
        result["load_s"] = time.monotonic() - t_launch
        if sampler.nvml:
            result["gpu_memory_used_mib_after_load"] = sampler.nvml.memory_used_mib(args.gpu_index)
        warm, control, _t0, _t1 = run_wave(server, arm, config["phone_layer_mask"], ["Warm-up: say hello."], 0, 1,
                                           args.warmup_tokens, run_tag + "-warm")
        result["warmup"] = {"errors": [request.error for request in warm if request.error],
                            "control_applied": control["applied"]}
        idle0 = time.monotonic()
        time.sleep(args.idle_s)
        idle = (idle0, time.monotonic())
        prompts = PROMPTS[:args.prompts]
        for concurrency in args.concurrency:
            raw = []
            n_waves = max(1, -(-len(prompts) // concurrency))
            for wave_index in range(n_waves):
                requests, control, t0, t1 = run_wave(server, arm, config["phone_layer_mask"], prompts, wave_index,
                                                     concurrency, args.n_predict, run_tag)
                raw.append((requests, control, t0, t1))
                periods = [bl.steady_period_ms(request.token_times, args.skip_tokens) for request in requests]
                periods = [value for value in periods if value]
                print("  %s c=%d wave %d/%d: %.1f s, step %.1f ms%s%s" % (
                    arm["name"], concurrency, wave_index + 1, n_waves, t1 - t0,
                    sorted(periods)[len(periods) // 2] if periods else float("nan"),
                    "" if control["applied"] in (True, None) else "  CONTROL FAILED",
                    "".join("  ERROR " + request.error for request in requests if request.error)), flush=True)
            levels.append((concurrency, raw))
        time.sleep(0.5)  # let the sampler pass the last window
    finally:
        result["exit_code"] = srv.stop()
        sampler.stop()
        for tap in taps:
            tap.close()
    result["server_tail"] = srv.tail
    if "idle" in locals():
        result["idle_window"] = sampler.window(*idle)
    summarized = []
    for concurrency, raw in levels:
        waves = [summarize_wave(requests, control, t0, t1, sampler, args.skip_tokens, args.gpu_index)
                 for requests, control, t0, t1 in raw]
        summarized.append({"concurrency": concurrency, "metrics": level_metrics(waves, concurrency, args.gpu_index),
                           "waves": waves})
    result["levels"] = summarized
    result["ffn"] = ffn_digest(srv.ffn)
    if taps:
        calls = [call for tap in taps for call in tap.calls]
        result["tap"] = {"summary": bl.summarize_calls(calls),
                         "by_helper": {tap.label: bl.summarize_calls(tap.calls) for tap in taps},
                         "errors": {tap.label: tap.errors for tap in taps}}
        with open(out_dir / ("tap-%s.jsonl" % arm["name"]), "w") as handle:
            for call in calls:
                handle.write(json.dumps(call) + "\n")
    result["host_after"] = host_contention()
    with open(out_dir / ("samples-%s.json" % arm["name"]), "w") as handle:
        json.dump({"gpu_index": args.gpu_index, "rows": sampler.rows}, handle)
    return result


def attach_identity(result, reference):
    if reference is None or reference is result:
        return
    ref = {}
    for level in reference.get("levels", []):
        for wave in level["waves"]:
            for row in wave["requests"]:
                ref[(level["concurrency"], row["wave"], row["slot"])] = row["token_ids"]
    for level in result.get("levels", []):
        compared = []
        for wave in level["waves"]:
            for row in wave["requests"]:
                comparison = bl.compare_tokens(ref.get((level["concurrency"], row["wave"], row["slot"])),
                                               row["token_ids"])
                row["identity_vs_cpu"] = comparison
                compared.append(comparison)
        known = [c for c in compared if c["identical"] is not None]
        level["metrics"]["identity_vs_cpu"] = {
            "compared": len(known), "identical": sum(1 for c in known if c["identical"]),
            "min_matching_prefix": min((c["matching_prefix"] for c in known), default=None)}


def summary_table(results):
    lines = ["| arm | c | step ms | ms/token | decode tok/s | e2e tok/s | GPU W | GPU J/tok | CPU J/tok | identical vs cpu |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, f="%.1f": "-" if v is None else f % v
    for result in results:
        for level in result.get("levels", []):
            m = level["metrics"]
            identity = m.get("identity_vs_cpu")
            ident = "-" if not identity else "%d/%d" % (identity["identical"], identity["compared"])
            lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                result["arm"]["name"], m["concurrency"], fmt(m["step_period_ms"]), fmt(m["ms_per_token"]),
                fmt(m["decode_tok_s"], "%.2f"), fmt(m["e2e_tok_s"], "%.2f"), fmt(m["gpu_decode_mean_w"]),
                fmt(m["gpu_decode_j_per_token"], "%.2f"), fmt(m["cpu_pkg_decode_j_per_token"], "%.2f"), ident))
    return "\n".join(lines)


def cmd_run(config, args):
    server = config["server"]
    binary = resolve_binary(server["binary"])
    arms = [bl.parse_arm(name, server["n_ff"], server["column_quantum"]) for name in args.arms.split(",")]
    if not os.path.isfile(binary):
        raise SystemExit("llama-server binary missing: " + binary)
    if not os.path.isfile(server["model"]):
        raise SystemExit("model missing: " + server["model"])
    if any(arm["helpers"] for arm in arms):
        has, where = library_has_ffn_split(binary)
        if not has:
            raise SystemExit("%s lacks the FFN split client: reconfigure with -DS41_SERVER_FFN_SPLIT=ON" % where)
        if not config["helpers"]:
            raise SystemExit("helper arms need phones[].workers in the config")
    out_dir = Path(args.out or (HERE / "results" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = "wifi%x" % (int(time.time()) & 0xFFFFFF)
    if args.verify_sha256:
        digest = sha256_file(server["model"])
        if digest != server["artifact_sha256"]:
            raise SystemExit("model sha256 %s != config %s" % (digest, server["artifact_sha256"]))
    with open(out_dir / "RUN.json", "w") as handle:
        json.dump({"config": config, "argv": sys.argv, "run_tag": run_tag,
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, handle, indent=1, default=str)
    results = []
    reference = None
    ref_path = out_dir / "RESULT-cpu.json"
    if ref_path.is_file():
        reference = json.load(open(ref_path))
    for arm in arms:
        print("== arm %s" % arm["name"], flush=True)
        try:
            result = run_arm(config, arm, args, out_dir, run_tag)
        except Exception as error:  # noqa: BLE001 - keep the other arms and record why this one failed
            print("  arm %s FAILED: %s" % (arm["name"], error), flush=True)
            result = {"schema": "wifi-server-arm-result-v1", "arm": arm, "run_tag": run_tag, "error": repr(error),
                      "levels": []}
        if arm["name"] == "cpu":
            reference = result
        attach_identity(result, reference)
        with open(out_dir / ("RESULT-%s.json" % arm["name"]), "w") as handle:
            json.dump(result, handle, indent=1)
        results.append(result)
        print(summary_table([result]), flush=True)
    table = summary_table(results)
    with open(out_dir / "SUMMARY.md", "w") as handle:
        handle.write("# server_bench %s\n\n%s\n" % (run_tag, table))
    print(table)
    print("results in", out_dir)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "probe", "plan"])
    parser.add_argument("--config", default=str(HERE / "wifi_config.json"))
    parser.add_argument("--arms", default="cpu")
    parser.add_argument("--concurrency", default=None, help="comma list, default from config workload [1,4]")
    parser.add_argument("--prompts", type=int, default=None)
    parser.add_argument("--n-predict", type=int, default=None)
    parser.add_argument("--skip-tokens", type=int, default=None, help="tokens excluded from the steady window")
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--idle-s", type=float, default=5.0, help="idle power window after load")
    parser.add_argument("--gpu-index", type=int, default=0, help="NVML index of the GPU the server uses")
    parser.add_argument("--out")
    parser.add_argument("--tap", action="store_true", help="route helpers through a local per-call timing proxy")
    parser.add_argument("--tap-base-port", type=int, default=7170)
    parser.add_argument("--host-override", help="use this host for every helper (127.0.0.1 = local stand-ins)")
    parser.add_argument("--no-preflight-probe", action="store_true")
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--model")
    parser.add_argument("--calls", type=int, default=200, help="probe: EXECUTE calls per row count")
    parser.add_argument("--gap-ms", type=float, default=8.0, help="probe: gap between calls (decode cadence)")
    args = parser.parse_args(argv)
    config = bl.load_config(args.config) if os.path.isfile(args.config) else bl.load_config({})
    if args.threads:
        config["server"]["threads"] = args.threads
    if args.port:
        config["server"]["port"] = args.port
    if args.model:
        config["server"]["model"] = args.model
    workload = config.get("workload", {})
    args.concurrency = [int(v) for v in (args.concurrency.split(",") if args.concurrency else
                                         workload.get("concurrency", [1, 4]))]
    args.prompts = args.prompts or workload.get("prompts", 5)
    args.n_predict = args.n_predict or workload.get("n_predict", 128)
    args.skip_tokens = args.skip_tokens if args.skip_tokens is not None else workload.get("skip_tokens", 8)
    if max(args.concurrency) > config["server"]["parallel"]:
        raise SystemExit("concurrency exceeds server.parallel")
    if max(args.concurrency) > config["server"]["max_tokens"] and args.arms != "cpu":
        print("warning: concurrency %d > max_tokens %d: decode steps with more rows run on the host"
              % (max(args.concurrency), config["server"]["max_tokens"]))
    return {"run": cmd_run, "probe": cmd_probe, "plan": cmd_plan}[args.command](config, args)


if __name__ == "__main__":
    sys.exit(main())
