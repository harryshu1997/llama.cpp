#!/usr/bin/env python3
"""Pure helpers shared by phone_workers.sh, server_bench.py and analyze.py.

Everything here is side-effect free (except the small GGUF header reader, which only reads) so it
can be unit tested without phones, GPUs or a server.  The wire formats mirror
examples/layersplit/ffn-split-protocol.h (protocol v6) and the llama-server FFN-split environment
parsed in tools/server/server.cpp (s41_server_ffn_runtime::init).

CLI (used by phone_workers.sh):
    python3 benchlib.py worker-commands --config CFG --action start|stop|status|echo-start|echo-stop|ip|wifi-tune|hash [--local]
prints one shell command per line (already quoted); phone_workers.sh runs or echoes them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import struct
import sys
from typing import Iterable

# ---------------------------------------------------------------------------------------------
# protocol v6 (examples/layersplit/ffn-split-protocol.h)
PROTOCOL_MAGIC = 0x46534631
PROTOCOL_VERSION = 6
FLAG_F16_IO = 1
FLAG_SWIGLU = 2
MSG_HELLO_REQUEST, MSG_HELLO_RESPONSE, MSG_EXECUTE_REQUEST, MSG_EXECUTE_RESPONSE = 1, 2, 3, 4
HELLO_REQUEST = struct.Struct("<IHHQIIHH32s4x")          # 64 B
HELLO_RESPONSE = struct.Struct("<IHHHHIIIIII4xQQIHH32s")  # 96 B
EXECUTE_REQUEST = struct.Struct("<IHHIiIIIII")           # 36 B
EXECUTE_RESPONSE = struct.Struct("<IHHHHIiIIIIIQ")       # 48 B

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def fnv1a32(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def artifact_digest(artifact_sha256: str) -> bytes:
    if not SHA256_RE.match(artifact_sha256 or ""):
        raise ValueError("artifact sha256 must be sha256:<64 hex>")
    return bytes.fromhex(artifact_sha256[7:])


def encode_hello(layer_mask: int, n_embd: int, max_columns: int, max_tokens: int,
                 artifact_sha256: str, f16_io: bool = True, swiglu: bool = True) -> bytes:
    flags = (FLAG_F16_IO if f16_io else 0) | (FLAG_SWIGLU if swiglu else 0)
    return HELLO_REQUEST.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, MSG_HELLO_REQUEST, layer_mask, n_embd,
                              max_columns, flags, max_tokens, artifact_digest(artifact_sha256))


def decode_hello_response(data: bytes) -> dict:
    (magic, version, message, status, flags, n_embd, n_ff, offset, max_columns, weight_type, layer_count,
     layer_mask, weight_hash, column_quantum, max_tokens, alternate32, artifact) = HELLO_RESPONSE.unpack(data)
    return {
        "ok": magic == PROTOCOL_MAGIC and version == PROTOCOL_VERSION and message == MSG_HELLO_RESPONSE
        and status == 0,
        "status": status, "flags": flags, "n_embd": n_embd, "n_ff": n_ff, "offset": offset,
        "max_columns": max_columns, "weight_type": weight_type, "layer_count": layer_count,
        "layer_mask": layer_mask, "weight_hash": "%016x" % weight_hash, "column_quantum": column_quantum,
        "max_tokens": max_tokens, "alternate_columns": alternate32 * 32,
        "artifact_sha256": "sha256:" + artifact.hex(),
    }


def encode_execute(request_id: int, layer: int, tokens: int, n_embd: int, columns: int,
                   payload: bytes, f16_io: bool = True, payload_hash: int | None = None) -> bytes:
    elements = n_embd * tokens
    if len(payload) != elements * (2 if f16_io else 4):
        raise ValueError("execute payload size differs from n_embd x tokens")
    digest = fnv1a32(payload) if payload_hash is None else payload_hash
    return EXECUTE_REQUEST.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, MSG_EXECUTE_REQUEST, request_id, layer,
                                elements, len(payload), digest, columns, tokens)


def decode_execute_request(data: bytes) -> dict:
    magic, version, message, request_id, layer, elements, payload_bytes, payload_hash, columns, tokens = \
        EXECUTE_REQUEST.unpack(data)
    return {"ok": magic == PROTOCOL_MAGIC and version == PROTOCOL_VERSION and message == MSG_EXECUTE_REQUEST,
            "request_id": request_id, "layer": layer, "elements": elements, "payload_bytes": payload_bytes,
            "payload_hash": payload_hash, "columns": columns, "tokens": tokens}


def decode_execute_response(data: bytes) -> dict:
    (magic, version, message, status, _reserved, request_id, layer, elements, payload_bytes, payload_hash,
     columns, tokens, compute_us) = EXECUTE_RESPONSE.unpack(data)
    return {"ok": magic == PROTOCOL_MAGIC and version == PROTOCOL_VERSION and message == MSG_EXECUTE_RESPONSE
            and status == 0, "status": status, "request_id": request_id, "layer": layer, "elements": elements,
            "payload_bytes": payload_bytes, "payload_hash": payload_hash, "columns": columns,
            "tokens": tokens, "compute_us": compute_us}


def f16_payload(values: Iterable[float]) -> bytes:
    values = list(values)
    return struct.pack("<%de" % len(values), *values)


def probe_payload(n_embd: int, tokens: int, seed: int = 1) -> bytes:
    """Deterministic small-magnitude f16 activations (an LCG, no numpy needed)."""
    state = seed & 0xFFFFFFFF
    out = []
    for _ in range(n_embd * tokens):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        out.append(((state >> 8) / float(1 << 23) - 0.5) * 0.25)
    return f16_payload(out)


# ---------------------------------------------------------------------------------------------
# layer masks, column shares, arms

def parse_layer_spec(text: str) -> int:
    """'0-5,7' -> mask (same grammar as parse_layer_spec in ffn-split-worker.cpp)."""
    if not text:
        raise ValueError("empty layer spec")
    mask = 0
    for item in text.split(","):
        if not item:
            raise ValueError("empty layer spec item")
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            if "-" in last_text:
                raise ValueError("bad layer range " + item)
            first, last = int(first_text), int(last_text)
        else:
            first = last = int(item)
        if first < 0 or last < first or last >= 64:
            raise ValueError("bad layer range " + item)
        for layer in range(first, last + 1):
            mask |= 1 << layer
    return mask


def layer_spec(mask: int) -> str:
    """mask -> compact 'a-b,c' spec."""
    if mask <= 0 or mask >= 1 << 64:
        raise ValueError("layer mask must be a nonempty 64-bit mask")
    layers = [layer for layer in range(64) if mask >> layer & 1]
    parts, start, prev = [], layers[0], layers[0]
    for layer in layers[1:] + [None]:
        if layer is not None and layer == prev + 1:
            prev = layer
            continue
        parts.append(str(start) if start == prev else "%d-%d" % (start, prev))
        if layer is not None:
            start = prev = layer
    return ",".join(parts)


def mask_layers(mask: int) -> list[int]:
    return [layer for layer in range(64) if mask >> layer & 1]


def lcm(a: int, b: int) -> int:
    return a // math.gcd(a, b) * b


def share_to_columns(share: float, n_ff: int, quantum: int) -> int:
    """Phone-owned FFN columns for a share in [0, 1]; must land exactly on the column quantum."""
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must be in [0, 1]")
    if quantum <= 0 or n_ff % quantum:
        raise ValueError("n_ff must be a multiple of the column quantum")
    exact = share * n_ff
    columns = int(round(exact / quantum)) * quantum
    if abs(columns - exact) > 0.5:
        raise ValueError("share %.4f x n_ff %d is not a multiple of the quantum %d (nearest %d)"
                         % (share, n_ff, quantum, columns))
    return columns


ARM_RE = re.compile(r"^(cpu|gpu|phone|cpu-helpers|split-(\d{1,3}))$")


def parse_arm(name: str, n_ff: int = 17408, quantum: int = 4352) -> dict:
    """Arm name -> {name, kind, share, columns, helpers}.

    cpu          no FFN runtime; the server CPU runs every CPU-resident layer (baseline)
    gpu          reference: every layer on the GPU (the model fits one A6000), no helpers
    cpu-helpers  FFN runtime configured with the helpers (deferred, never activated): overhead control
    phone        phones own 100 % of the columns of their layers (the desktop mode)
    split-NN     phones own NN % of the columns (trailing suffix), the CPU the leading rest, concurrently
    """
    match = ARM_RE.match(name)
    if not match:
        raise ValueError("unknown arm " + repr(name))
    if name in ("cpu", "gpu"):
        return {"name": name, "kind": name, "share": 0.0, "columns": 0, "helpers": False}
    if name == "cpu-helpers":
        return {"name": name, "kind": "cpu-helpers", "share": 0.0, "columns": 0, "helpers": True}
    share = 1.0 if name == "phone" else int(match.group(2)) / 100.0
    if share <= 0.0 or share > 1.0:
        raise ValueError("split share must be in (0, 100]")
    columns = share_to_columns(share, n_ff, quantum)
    return {"name": name, "kind": "phone" if columns == n_ff else "split", "share": columns / n_ff,
            "columns": columns, "helpers": True}


def policy_hash(layer_mask: int, columns: int, arm: str) -> str:
    text = json.dumps({"arm": arm, "columns": columns, "layer_mask": layer_mask}, sort_keys=True)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------------------------
# configuration

DEFAULT_SERVER = {
    "binary": "build-cuda-s43/bin/llama-server",
    "worker_binary": "build-cuda-s43/bin/llama-ffn-split-worker",
    "model": "/home/myid/zs89458/Documents/models/Qwen3-14B-Q4KM-dequant-f16.gguf",
    "artifact_sha256": "sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718",
    "alias": "qwen3-14b-q4km-dequant-f16",
    "cuda_visible_devices": "0",
    "gpu_layers": 16,
    "threads": 48,
    "threads_batch": 64,
    "ctx_size": 4096,
    "parallel": 4,
    "batch_size": 2048,
    "ubatch_size": 512,
    "host": "127.0.0.1",
    "port": 18620,
    "n_embd": 5120,
    "n_ff": 17408,
    "activation": "swiglu",
    "max_tokens": 4,
    "column_quantum": 4352,
    "timeout_ms": 120000,
    "cuda_graphs": False,
    "extra_args": [],
}


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = here
    while root != "/" and not os.path.isfile(os.path.join(root, "tools", "server", "server.cpp")):
        root = os.path.dirname(root)
    return root if root != "/" else here


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("config: " + message)


def load_config(source) -> dict:
    """Parse + validate the harness config (path or dict). Returns a normalized copy."""
    if isinstance(source, str):
        with open(source) as handle:
            raw = json.load(handle)
    else:
        raw = json.loads(json.dumps(source))
    _check(isinstance(raw, dict), "top level must be an object")
    server = dict(DEFAULT_SERVER)
    server.update(raw.get("server", {}))
    for key in ("gpu_layers", "threads", "threads_batch", "ctx_size", "parallel", "batch_size", "ubatch_size",
                "port", "n_embd", "n_ff", "max_tokens", "column_quantum", "timeout_ms"):
        _check(type(server[key]) is int and server[key] >= 0, "server.%s must be a non-negative int" % key)
    _check(SHA256_RE.match(server["artifact_sha256"]) is not None, "server.artifact_sha256 is invalid")
    _check(server["n_ff"] % server["column_quantum"] == 0, "server.column_quantum must divide n_ff")
    _check(server["activation"] in ("swiglu", "geglu"), "server.activation must be swiglu or geglu")
    phones = raw.get("phones", [])
    _check(isinstance(phones, list), "phones must be a list")
    helpers, labels, covered, endpoints = [], set(), 0, set()
    for phone in phones:
        _check(isinstance(phone, dict) and phone.get("name"), "every phone needs a name")
        _check(isinstance(phone.get("adb", []), list), "phone.adb must be an argv list")
        _check(not phone.get("lock_path") or len(phone.get("workers", [])) <= 1,
               "phone %s: lock_path (one kernel lock per phone) allows a single worker" % phone["name"])
        for worker in phone.get("workers", []):
            label = worker.get("label", "")
            _check(re.fullmatch(r"[A-Za-z0-9_-]{1,32}", label or "") is not None,
                   "worker label %r must be 1-32 [A-Za-z0-9_-]" % label)
            _check(label not in labels, "duplicate worker label " + label)
            labels.add(label)
            mask = parse_layer_spec(worker["layers"])
            _check(mask & covered == 0, "worker %s overlaps another worker's layers" % label)
            covered |= mask
            port = worker.get("port")
            _check(type(port) is int and 0 < port < 65536, "worker %s port is invalid" % label)
            host = worker.get("host", phone.get("wlan_ip", ""))
            _check(bool(host), "worker %s has no host (phone.wlan_ip)" % label)
            _check((host, port) not in endpoints, "duplicate endpoint %s:%d" % (host, port))
            endpoints.add((host, port))
            quantum = worker.get("column_quantum", phone.get("column_quantum", server["column_quantum"]))
            columns = worker.get("columns", phone.get("columns", server["n_ff"]))
            max_tokens = worker.get("max_tokens", phone.get("max_tokens", server["max_tokens"]))
            _check(columns == server["n_ff"], "worker %s must serve the full width %d (server columns)"
                   % (label, server["n_ff"]))
            _check(max_tokens == server["max_tokens"], "worker %s max_tokens must equal server.max_tokens" % label)
            _check(server["n_ff"] % quantum == 0, "worker %s column quantum must divide n_ff" % label)
            helpers.append({"label": label, "phone": phone["name"], "host": host, "port": port,
                            "layer_mask": mask, "layers": layer_spec(mask), "backend": worker.get("backend", "CPU"),
                            "column_quantum": quantum, "columns": columns, "max_tokens": max_tokens})
    _check(len(helpers) <= 8, "llama-server accepts at most 8 FFN helpers (server.cpp S41_SERVER_FFN_HELPERS)")
    union_quantum = 1
    for helper in helpers:
        union_quantum = lcm(union_quantum, helper["column_quantum"])
    if helpers:
        _check(server["column_quantum"] % union_quantum == 0,
               "server.column_quantum %d must be a multiple of the helpers' LCM quantum %d"
               % (server["column_quantum"], union_quantum))
    cpu_resident = raw.get("cpu_resident_layers")
    return {"server": server, "phones": phones, "helpers": helpers, "phone_layer_mask": covered,
            "union_quantum": union_quantum, "cpu_resident_layers": cpu_resident,
            "workload": raw.get("workload", {}), "model_physics": raw.get("model_physics", {})}


def helpers_for_tap(helpers: list[dict], tap_base_port: int) -> list[dict]:
    """Point every helper at a local tap proxy port; the proxy forwards to the real host:port."""
    out = []
    for index, helper in enumerate(helpers):
        row = dict(helper)
        row["target_host"], row["target_port"] = helper["host"], helper["port"]
        row["host"], row["port"] = "127.0.0.1", tap_base_port + index
        out.append(row)
    return out


def server_ffn_environment(config: dict, arm: dict, helpers: list[dict] | None = None) -> dict:
    """The S41_SERVER_FFN_* launch environment (tools/server/server.cpp:196-300) for a helper arm.

    Runtime control (decode-boundary protocol) as on the desktop: connections are deferred, the policy
    is posted per slot after prefill through POST /v1/chat/completions/control, so the phones only
    see decode rows (<= max_tokens) exactly like the qualified desktop runs.
    """
    if not arm["helpers"]:
        return {}
    server = config["server"]
    helpers = config["helpers"] if helpers is None else helpers
    if not helpers:
        raise ValueError("arm %s needs FFN helpers in the config" % arm["name"])
    union = 0
    for helper in helpers:
        union |= helper["layer_mask"]
    env = {
        "S41_SERVER_FFN_ARTIFACT_SHA256": server["artifact_sha256"],
        "S41_SERVER_FFN_N_EMBD": str(server["n_embd"]),
        "S41_SERVER_FFN_LAYER_MASK": str(union),
        "S41_SERVER_FFN_COLUMNS": str(server["n_ff"]),
        "S41_SERVER_FFN_F16_IO": "1",
        "S41_SERVER_FFN_ACTIVATION": server["activation"],
        "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
        "S41_SERVER_FFN_MAX_TOKENS": str(server["max_tokens"]),
        "S41_SERVER_FFN_TIMEOUT_MS": str(server["timeout_ms"]),
    }
    if len(helpers) == 1:
        env.update({"S41_SERVER_FFN_TRANSPORT": "tcp", "S41_SERVER_FFN_HOST": helpers[0]["host"],
                    "S41_SERVER_FFN_PORT": str(helpers[0]["port"])})
        return env
    env["S41_SERVER_FFN_HELPERS"] = str(len(helpers))
    for index, helper in enumerate(helpers):
        prefix = "S41_SERVER_FFN_HELPER%d_" % index
        env[prefix + "LABEL"] = helper["label"]
        env[prefix + "LAYER_MASK"] = str(helper["layer_mask"])
        env[prefix + "TRANSPORT"] = "tcp"
        env[prefix + "HOST"] = helper["host"]
        env[prefix + "PORT"] = str(helper["port"])
    return env


def server_argv(config: dict, arm: dict, binary: str | None = None) -> list[str]:
    """llama-server argv; mirrors research_dev/scheduler/adapters/llama_server.py:804-862."""
    server = config["server"]
    gpu_layers = 999 if arm["kind"] == "gpu" else server["gpu_layers"]
    argv = [binary or server["binary"], "--model", server["model"], "--alias", server["alias"],
            "--fit", "off", "--ctx-size", str(server["ctx_size"]), "--parallel", str(server["parallel"]),
            "--batch-size", str(server["batch_size"]), "--ubatch-size", str(server["ubatch_size"]),
            "--flash-attn", "on", "--cont-batching", "--kv-unified", "--no-cache-idle-slots",
            "--cache-type-k", "f16", "--cache-type-v", "f16", "--split-mode", "none",
            "--n-gpu-layers", str(gpu_layers), "--main-gpu", "0", "--host", server["host"],
            "--port", str(server["port"]), "--metrics", "--slots", "--no-webui", "--log-colors", "off",
            "--log-timestamps", "--threads", str(server["threads"]), "--threads-batch",
            str(server["threads_batch"])]
    if gpu_layers > 0:
        argv += ["--device", "CUDA0"]
    return argv + [str(value) for value in server.get("extra_args", [])]


def server_process_environment(config: dict, arm: dict, helpers: list[dict] | None = None,
                               base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key.startswith(("S41_SERVER_FFN", "LLAMA_FFN_SPLIT", "S41_SERVER_LOGITS")):
            del env[key]
    env["CUDA_VISIBLE_DEVICES"] = config["server"]["cuda_visible_devices"]
    if not config["server"].get("cuda_graphs"):
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    env.update(server_ffn_environment(config, arm, helpers))
    return env


# ---------------------------------------------------------------------------------------------
# server stderr parsing

_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\S*)")
_NUMERIC_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d*(e[-+]?\d+)?$")


def _kv(text: str) -> dict:
    out = {}
    for key, value in _KV_RE.findall(text):
        if _NUMERIC_RE.match(value):
            out[key] = int(value)
        elif _FLOAT_RE.match(value):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def parse_ffn_line(line: str) -> dict | None:
    """Classify one llama-server stderr line of the FFN split; None for unrelated lines.

    Kinds: helper, caps, ready, usb_call (per-call timing, FunctionFS only), call (per-call proof,
    every transport), summary (shutdown JSON per helper), shape, error, reset, control (FFNCONTROL).
    """
    line = line.rstrip("\n")
    index = line.find("S41SERVERFFN")
    if index < 0:
        index = line.find("FFNCONTROL ")
        if index < 0:
            return None
        row = _kv(line[index + len("FFNCONTROL "):])
        row["kind"] = "control"
        return row
    body = line[index:]
    tag, _, rest = body.partition(" ")
    if tag == "S41SERVERFFN" and rest.startswith("{"):
        try:
            row = json.loads(rest)
        except ValueError:
            return {"kind": "unparsed", "text": body}
        row["kind"] = "summary"
        return row
    if tag == "S41SERVERFFNSHAPE" and rest.startswith("{"):
        row = json.loads(rest)
        row["kind"] = "shape"
        return row
    kinds = {"S41SERVERFFNHELPER": "helper", "S41SERVERFFNCAPS": "caps", "S41SERVERFFNUSB": "usb_call",
             "S41SERVERFFNCALL": "call", "S41SERVERFFNERROR": "error", "S41SERVERFFNRESET": "reset"}
    if tag in kinds:
        row = _kv(rest)
        row["kind"] = kinds[tag]
        if tag == "S41SERVERFFNERROR":
            row["text"] = rest
        if tag == "S41SERVERFFNUSB":
            row.update(usb_call_timing(row))
        return row
    if tag == "S41SERVERFFN" and rest.startswith("ready"):
        row = _kv(rest)
        row["kind"] = "ready"
        return row
    if tag == "S41SERVERFFN":
        return {"kind": "info", "text": rest}
    return None


def usb_call_timing(row: dict) -> dict:
    """Derived per-call times of an S41SERVERFFNUSB line (ms)."""
    try:
        started, h2d, d2h = row["started_ns"], row["h2d_completed_ns"], row["d2h_completed_ns"]
        compute_ms = row["compute_us"] / 1000.0
    except KeyError:
        return {}
    rpc_ms = (d2h - started) / 1e6
    return {"h2d_ms": (h2d - started) / 1e6, "rpc_ms": rpc_ms, "compute_ms": compute_ms,
            "transport_ms": rpc_ms - compute_ms}


def summarize_calls(rows: list[dict]) -> dict:
    """Aggregate per-call rows (usb_call or tap rows with rpc_ms/compute_ms) -> percentiles."""
    rpc = [row["rpc_ms"] for row in rows if "rpc_ms" in row]
    compute = [row["compute_ms"] for row in rows if "compute_ms" in row]
    transport = [row["transport_ms"] for row in rows if "transport_ms" in row]
    out = {"calls": len(rows)}
    for name, values in (("rpc_ms", rpc), ("compute_ms", compute), ("transport_ms", transport)):
        out[name] = distribution(values)
    by_layer = {}
    for row in rows:
        by_layer.setdefault(row.get("layer"), []).append(row.get("rpc_ms"))
    out["rpc_p50_ms_by_layer"] = {str(layer): percentile([v for v in values if v is not None], 0.5)
                                  for layer, values in sorted(by_layer.items(), key=lambda kv: (kv[0] is None, kv[0]))}
    return out


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank on sorted values (q in [0, 1]); None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))]


def distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p90": None, "p99": None, "p999": None, "max": None, "mean": None}
    return {"n": len(values), "p50": percentile(values, 0.5), "p90": percentile(values, 0.9),
            "p99": percentile(values, 0.99), "p999": percentile(values, 0.999), "max": max(values),
            "mean": sum(values) / len(values)}


# ---------------------------------------------------------------------------------------------
# energy counters

def counter_delta(first: int, last: int, wrap: int | None) -> int:
    """Monotonic energy counter delta with optional wrap-around (RAPL max_energy_range_uj)."""
    if last >= first:
        return last - first
    if not wrap:
        raise ValueError("energy counter went backwards without a wrap range")
    return wrap - first + last


def energy_between(samples: list[tuple[float, float]], t0: float, t1: float) -> float | None:
    """Linear interpolation of a cumulative energy series [(t, joules)] between t0 and t1."""
    if len(samples) < 2 or t1 <= t0 or t0 < samples[0][0] or t1 > samples[-1][0]:
        return None

    def at(t):
        for (ta, ea), (tb, eb) in zip(samples, samples[1:]):
            if ta <= t <= tb:
                return ea if tb == ta else ea + (eb - ea) * (t - ta) / (tb - ta)
        return None
    first, last = at(t0), at(t1)
    return None if first is None or last is None else last - first


def mean_between(samples: list[tuple[float, float]], t0: float, t1: float) -> float | None:
    values = [value for t, value in samples if t0 <= t <= t1]
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------------------------------
# request timing

def steady_period_ms(token_times: list[float], skip: int) -> float | None:
    """Mean inter-token period (ms) over tokens [skip, N) of one streamed request (times in s)."""
    if len(token_times) - skip < 2:
        return None
    window = token_times[skip:]
    return (window[-1] - window[0]) * 1000.0 / (len(window) - 1)


def compare_tokens(reference: list[int] | None, candidate: list[int] | None) -> dict:
    if reference is None or candidate is None:
        return {"identical": None, "first_divergence": None, "matching_prefix": None}
    prefix = 0
    for a, b in zip(reference, candidate):
        if a != b:
            break
        prefix += 1
    identical = prefix == len(reference) == len(candidate)
    return {"identical": identical, "first_divergence": None if identical else prefix, "matching_prefix": prefix}


# ---------------------------------------------------------------------------------------------
# GGUF tensor sizes (header only)

GGML_TYPE_SIZE = {  # type id: (block elements, bytes per block)
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
    10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    30: (1, 2),
}


def tensor_nbytes(dims: list[int], ggml_type: int) -> int:
    block, size = GGML_TYPE_SIZE[ggml_type]
    elements = 1
    for value in dims:
        elements *= value
    if elements % block:
        raise ValueError("tensor elements are not a multiple of the block size")
    return elements // block * size


def read_gguf_tensors(path: str) -> tuple[dict, dict]:
    """(metadata subset, {tensor name: nbytes}) from a GGUF header; skips big arrays."""
    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    with open(path, "rb") as handle:
        def read(fmt):
            size = struct.calcsize(fmt)
            return struct.unpack(fmt, handle.read(size))[0]

        def read_string():
            length = read("<Q")
            return handle.read(length).decode("utf-8", "replace")

        def read_value(kind):
            if kind in scalar:
                return read(scalar[kind])
            if kind == 8:
                return read_string()
            if kind == 9:
                item_kind, count = read("<I"), read("<Q")
                if item_kind in scalar:
                    size = struct.calcsize(scalar[item_kind])
                    if count > 64:
                        handle.seek(size * count, 1)
                        return None
                    return [read(scalar[item_kind]) for _ in range(count)]
                values = [read_value(item_kind) for _ in range(count)]
                return None if count > 64 else values
            raise ValueError("unknown GGUF value type %d" % kind)

        if handle.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = read("<I")
        tensor_count, kv_count = read("<Q"), read("<Q")
        metadata = {"gguf_version": version}
        for _ in range(kv_count):
            key = read_string()
            value = read_value(read("<I"))
            if value is not None and not isinstance(value, list):
                metadata[key] = value
        tensors = {}
        for _ in range(tensor_count):
            name = read_string()
            dims = [read("<Q") for _ in range(read("<I"))]
            ggml_type = read("<I")
            read("<Q")
            tensors[name] = tensor_nbytes(dims, ggml_type)
    return metadata, tensors


def layer_byte_table(tensors: dict) -> dict:
    """{'layers': {i: {'ffn': B, 'attn': B, 'other': B}}, 'output': B, 'token_embd': B}."""
    layers = {}
    for name, size in tensors.items():
        match = re.match(r"^blk\.(\d+)\.(.+)$", name)
        if not match:
            continue
        layer, rest = int(match.group(1)), match.group(2)
        row = layers.setdefault(layer, {"ffn": 0, "attn": 0, "other": 0})
        if rest.startswith(("ffn_gate.", "ffn_up.", "ffn_down.")):
            row["ffn"] += size
        elif rest.startswith("attn_") and not rest.startswith(("attn_norm", "attn_q_norm", "attn_k_norm")):
            row["attn"] += size
        else:
            row["other"] += size
    return {"layers": layers, "output": tensors.get("output.weight", 0),
            "token_embd": tensors.get("token_embd.weight", 0)}


QWEN3_14B_F16_BYTES = {  # fallback when the GGUF is not readable: Qwen3-14B dims, f16 weights
    "n_layer": 40, "ffn": 3 * 5120 * 17408 * 2, "attn": (2 * 5120 * 5120 + 2 * 5120 * 1024) * 2,
    "other": (2 * 5120 + 2 * 128) * 4, "output": 5120 * 151936 * 2,
}


def model_bytes(path: str | None) -> dict:
    """Per-layer byte table from the model GGUF, or the Qwen3-14B f16 fallback."""
    if path and os.path.isfile(path):
        try:
            metadata, tensors = read_gguf_tensors(path)
            table = layer_byte_table(tensors)
            table["source"] = path
            arch = metadata.get("general.architecture", "")
            table["n_layer"] = metadata.get(arch + ".block_count", len(table["layers"]))
            table["n_embd"] = metadata.get(arch + ".embedding_length")
            table["n_ff"] = metadata.get(arch + ".feed_forward_length")
            return table
        except (OSError, ValueError, KeyError, struct.error):
            pass
    fallback = QWEN3_14B_F16_BYTES
    return {"source": "qwen3-14b-f16-dims", "n_layer": fallback["n_layer"], "n_embd": 5120, "n_ff": 17408,
            "layers": {layer: {"ffn": fallback["ffn"], "attn": fallback["attn"], "other": fallback["other"]}
                       for layer in range(fallback["n_layer"])},
            "output": fallback["output"], "token_embd": 5120 * 151936 * 2}


# ---------------------------------------------------------------------------------------------
# time model

def cpu_resident_layers(n_layer: int, gpu_layers: int) -> list[int]:
    """llama.cpp -ngl N offloads the LAST N repeating layers; the output head only when N > n_layer."""
    return list(range(max(0, n_layer - gpu_layers)))


def predict_cpu_resident_ms(table: dict, cpu_layers: list[int], share: float, owner_of: dict,
                            bw_cpu_gbs: float, phones: dict, model: str = "overlap",
                            output_on_cpu: bool = True) -> dict:
    """Predicted CPU-resident time per decode step (ms).

    table     layer byte table (layer_byte_table / model_bytes)
    share     phone-owned fraction of every phone layer's FFN columns (0 = cpu arm)
    owner_of  {layer: phone name} for phone-owned layers
    phones    {name: {"bw_gbs": effective FFN streaming rate, "rtt_ms": network round trip per call}}
    model     "overlap"   per layer: attn + max(CPU (1-share) FFN, owner phone share FFN + RTT). This is
                          what llama-server does: every layer has exactly ONE owner helper (disjoint masks,
                          tools/server/server.cpp:258-272) and the host partial runs concurrently with it.
              "aggregate" PLAN physics: fixed + max((1-share) F / BW_cpu, share F / sum(BW_phones)) +
                          calls x RTT, i.e. every phone streams its part of EVERY layer concurrently with
                          the CPU (a 3-way column split the server does not implement). At the balanced
                          share this is bytes / (BW_cpu + sum BW_phones) + calls x RTT.
    Attention, norms and the output head (when not offloaded) always stream on the CPU.
    """
    to_ms = lambda nbytes, gbs: nbytes / (gbs * 1e9) * 1e3
    fixed, ffn_total, calls, rtt_total, per_layer = 0.0, 0.0, 0, 0.0, 0.0
    owners = set()
    for layer in cpu_layers:
        row = table["layers"][layer]
        fixed += to_ms(row["attn"] + row["other"], bw_cpu_gbs)
        ffn = row["ffn"]
        ffn_total += ffn
        owner = owner_of.get(layer)
        if share > 0 and owner is not None:
            spec = phones[owner]
            owners.add(owner)
            calls += 1
            rtt_total += spec["rtt_ms"]
            host_ms = to_ms(ffn * (1.0 - share), bw_cpu_gbs)
            phone_ms = to_ms(ffn * share, spec["bw_gbs"]) + spec["rtt_ms"]
            per_layer += max(host_ms, phone_ms)
        else:
            per_layer += to_ms(ffn, bw_cpu_gbs)
    output_ms = to_ms(table["output"], bw_cpu_gbs) if output_on_cpu else 0.0
    if model == "overlap":
        ffn_ms = per_layer
    elif model == "aggregate":
        if calls:
            phones_bw = sum(phones[name]["bw_gbs"] for name in owners)
            ffn_ms = max(to_ms(ffn_total * (1.0 - share), bw_cpu_gbs), to_ms(ffn_total * share, phones_bw))
            ffn_ms += rtt_total
        else:
            ffn_ms = to_ms(ffn_total, bw_cpu_gbs)
    else:
        raise ValueError("unknown model " + model)
    return {"model": model, "ms": fixed + ffn_ms + output_ms, "fixed_ms": fixed, "ffn_ms": ffn_ms,
            "output_ms": output_ms, "calls": calls, "rtt_total_ms": rtt_total}


def best_share(table: dict, cpu_layers: list[int], owner_of: dict, bw_cpu_gbs: float, phones: dict,
               quantum_share: float = 0.25) -> tuple[float, float]:
    """Share (multiple of quantum_share) minimizing the overlap model."""
    best = (0.0, predict_cpu_resident_ms(table, cpu_layers, 0.0, owner_of, bw_cpu_gbs, phones)["ms"])
    steps = int(round(1.0 / quantum_share))
    for step in range(1, steps + 1):
        share = step * quantum_share
        value = predict_cpu_resident_ms(table, cpu_layers, share, owner_of, bw_cpu_gbs, phones)["ms"]
        if value < best[1]:
            best = (share, value)
    return best


# ---------------------------------------------------------------------------------------------
# phone worker commands (consumed by phone_workers.sh)

def _env_words(environment: dict) -> list[str]:
    return ["%s=%s" % (key, value) for key, value in sorted(environment.items())]


def worker_phone_argv(phone: dict, worker: dict, server: dict) -> list[str]:
    """argv of one llama-ffn-split-worker in TCP mode on the phone (bound to its WLAN side)."""
    mask = parse_layer_spec(worker["layers"])
    libraries = worker.get("library_dirs", phone.get("library_dirs", []))
    environment = dict(phone.get("environment", {}))
    environment.update(worker.get("environment", {}))
    if libraries:
        environment.setdefault("LD_LIBRARY_PATH", ":".join(libraries))
    argv = ["env"] + _env_words(environment) + [
        worker.get("worker_binary", phone.get("worker_binary")),
        "-m", worker.get("model", phone.get("model")),
        "--artifact-sha256", server["artifact_sha256"],
        "--layers", layer_spec(mask),
        "--columns", str(worker.get("columns", phone.get("columns", server["n_ff"]))),
        "--column-quantum", str(worker.get("column_quantum", phone.get("column_quantum", server["column_quantum"]))),
        "--backend", worker.get("backend", "CPU"),
        "--port", str(worker["port"]),
        "--bind", worker.get("bind", phone.get("bind", "0.0.0.0")),
        "--f16-io",
        "--max-tokens", str(worker.get("max_tokens", phone.get("max_tokens", server["max_tokens"]))),
        "--max-requests", str(worker.get("max_requests", phone.get("max_requests", 0))),
    ]
    if any(word is None for word in argv):
        raise ValueError("worker %s lacks worker_binary or model" % worker.get("label"))
    return argv


SAFE_WORD = re.compile(r"^[A-Za-z0-9@%+=:,./_-]+$")


def _words(words: list[str]) -> str:
    """Join phone-side words; they must not need quoting (paths/env values are plain)."""
    for word in words:
        if not SAFE_WORD.match(str(word)):
            raise ValueError("phone-side word needs quoting: %r" % (word,))
    return " ".join(str(word) for word in words)


def _dq(text: str) -> str:
    """Escape for a bash double-quoted string."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")


def phone_shell_command(phone: dict, script: str) -> str:
    """Local shell command: <adb argv> shell "su -c '<script>'" (script uses no single quotes)."""
    if "'" in script:
        raise ValueError("phone scripts must not contain single quotes")
    remote = script
    if phone.get("as_root", True):
        remote = "%s '%s'" % (phone.get("su", "su -c"), script)
    return shlex.join(list(phone.get("adb", ["adb"])) + ["shell", "-T"]) + ' "' + _dq(remote) + '"'


def phone_log_dir(phone: dict) -> str:
    return phone.get("log_dir", "/data/local/tmp/wifi-server-20260928")


def worker_start_script(phone: dict, worker: dict, server: dict) -> str:
    log_dir = phone_log_dir(phone)
    label = worker["label"]
    command = _words(worker_phone_argv(phone, worker, server))
    log, pid = "%s/%s.log" % (log_dir, label), "%s/%s.pid" % (log_dir, label)
    lock = phone.get("lock_path")
    if lock:
        # hold the kernel lock for the worker's lifetime (the Pixel's qualified launch does this)
        command = 'sh -c "exec 9>%s; flock -n 9 || exit 73; exec %s"' % (_words([lock]), command)
    ready_wait = int(worker.get("ready_timeout_s", phone.get("ready_timeout_s", 480)))
    return ("mkdir -p {d} && cd {d} && if [ -f {pid} ] && kill -0 $(cat {pid}) 2>/dev/null; then "
            "echo {label} already running; else rm -f {log}; "
            "nohup {setsid}{command} > {log} 2>&1 < /dev/null & echo $! > {pid}; fi; "
            "i=0; while [ $i -lt {wait} ]; do grep -q \"ready backend=\" {log} && break; "
            "kill -0 $(cat {pid}) 2>/dev/null || break; sleep 1; i=$((i+1)); done; "
            "grep \"ready backend=\" {log} || {{ echo {label} NOT READY; tail -n 40 {log}; exit 1; }}"
            ).format(d=log_dir, pid=pid, log=log, command=command, label=label, wait=ready_wait,
                     setsid="setsid " if phone.get("setsid", True) else "")


def worker_stop_script(phone: dict, worker: dict) -> str:
    pid = "%s/%s.pid" % (phone_log_dir(phone), worker["label"])
    return ("if [ -f {pid} ]; then kill -TERM $(cat {pid}) 2>/dev/null; sleep 1; "
            "if kill -0 $(cat {pid}) 2>/dev/null; then echo {label} still alive, a client may be connected; "
            "else rm -f {pid}; echo {label} stopped; fi; else echo {label} not running; fi"
            ).format(pid=pid, label=worker["label"])


def worker_status_script(phone: dict, worker: dict) -> str:
    log_dir = phone_log_dir(phone)
    label = worker["label"]
    pid, log = "%s/%s.pid" % (log_dir, label), "%s/%s.log" % (log_dir, label)
    return ("if [ -f {pid} ] && kill -0 $(cat {pid}) 2>/dev/null; then echo {label} running pid $(cat {pid}); "
            "else echo {label} not running; fi; grep -E \"ready backend=|client (dis)?connected|requests=\" {log} "
            "2>/dev/null | tail -n 4").format(pid=pid, log=log, label=label)


def echo_start_script(phone: dict) -> str:
    port = int(phone.get("echo_port", 7070))
    pid = "%s/echo-%d.pid" % (phone_log_dir(phone), port)
    return ("mkdir -p {d}; nohup setsid toybox nc -L -p {port} cat > /dev/null 2>&1 < /dev/null & echo $! > {pid}; "
            "sleep 0.5; kill -0 $(cat {pid}) && echo echo server on :{port} pid $(cat {pid})"
            ).format(d=phone_log_dir(phone), port=port, pid=pid)


def echo_stop_script(phone: dict) -> str:
    port = int(phone.get("echo_port", 7070))
    pid = "%s/echo-%d.pid" % (phone_log_dir(phone), port)
    return "kill -TERM $(cat {pid}) 2>/dev/null; rm -f {pid}; echo echo server :{port} stopped".format(
        pid=pid, port=port)


WIFI_TUNE_SCRIPT = ("svc power stayon true; cmd wifi force-low-latency-mode enabled; "
                    "cmd wifi force-hi-perf-mode enabled; "
                    "(command -v iw >/dev/null && iw dev wlan0 set power_save off) || true; "
                    "ip -4 -brief addr show wlan0")


def local_worker_argv(config: dict, worker: dict, binary: str, phone: dict | None = None) -> list[str]:
    """Stand-in: the same worker built for this host, on 127.0.0.1, CPU backend, full model."""
    server = config["server"]
    quantum = worker.get("column_quantum", (phone or {}).get("column_quantum", server["column_quantum"]))
    return [binary, "-m", server["model"], "--artifact-sha256", server["artifact_sha256"],
            "--layers", layer_spec(parse_layer_spec(worker["layers"])), "--columns", str(server["n_ff"]),
            "--column-quantum", str(quantum),
            "--backend", "CPU", "--port", str(worker["port"]), "--bind", "127.0.0.1", "--f16-io",
            "--max-tokens", str(server["max_tokens"]), "--max-requests", "0"]


def worker_commands(config: dict, action: str, local: bool = False, local_dir: str = "") -> list[str]:
    server = config["server"]
    lines = []
    for phone in config["phones"]:
        workers = phone.get("workers", [])
        if local:
            if action not in ("start", "stop", "status"):
                continue
            root = repo_root()
            binary = server["worker_binary"]
            binary = binary if os.path.isabs(binary) else os.path.join(root, binary)
            cpus = phone.get("local_cpus", "96-127")
            for worker in workers:
                label = worker["label"]
                log = os.path.join(local_dir, label + ".log")
                pid = os.path.join(local_dir, label + ".pid")
                if action == "start":
                    argv = ["taskset", "-c", cpus] + local_worker_argv(config, worker, binary, phone)
                    # mkdir must not share the backgrounded list: a backgrounded `a && b &` subshell keeps
                    # the caller's stdout open for the worker's lifetime
                    lines.append("mkdir -p %s; CUDA_VISIBLE_DEVICES= nohup %s > %s 2>&1 < /dev/null & echo $! > %s; "
                                 "for i in $(seq 1 600); do grep -q 'ready backend=' %s && break; "
                                 "kill -0 $(cat %s) 2>/dev/null || break; sleep 1; done; grep 'ready backend=' %s "
                                 "|| { echo '%s NOT READY'; tail -n 20 %s; }"
                                 % (shlex.quote(local_dir), shlex.join(argv), shlex.quote(log), shlex.quote(pid),
                                    shlex.quote(log), shlex.quote(pid), shlex.quote(log), label, shlex.quote(log)))
                elif action == "stop":
                    lines.append("[ -f %s ] && kill -TERM $(cat %s) 2>/dev/null; rm -f %s; echo '%s stopped'"
                                 % (shlex.quote(pid), shlex.quote(pid), shlex.quote(pid), label))
                else:
                    lines.append("[ -f %s ] && kill -0 $(cat %s) 2>/dev/null && echo '%s running' || echo '%s not running'"
                                 % (shlex.quote(pid), shlex.quote(pid), label, label))
            continue
        if action == "start":
            scripts = [worker_start_script(phone, worker, server) for worker in workers]
        elif action == "stop":
            scripts = [worker_stop_script(phone, worker) for worker in workers]
        elif action == "status":
            scripts = [worker_status_script(phone, worker) for worker in workers]
        elif action == "echo-start":
            scripts = [echo_start_script(phone)]
        elif action == "echo-stop":
            scripts = [echo_stop_script(phone)]
        elif action == "ip":
            scripts = ["ip -4 -brief addr show wlan0"]
        elif action == "wifi-tune":
            scripts = [WIFI_TUNE_SCRIPT]
        elif action == "hash":
            paths = sorted({path for worker in workers for path in (
                worker.get("worker_binary", phone.get("worker_binary")), worker.get("model", phone.get("model")))
                if path})
            scripts = ["sha256sum " + _words(paths)]
        else:
            raise ValueError("unknown action " + action)
        for script in scripts:
            lines.append(phone_shell_command(phone, script))
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = sub.add_parser("worker-commands")
    commands.add_argument("--config", required=True)
    commands.add_argument("--action", required=True,
                          choices=["start", "stop", "status", "echo-start", "echo-stop", "ip", "wifi-tune", "hash"])
    commands.add_argument("--local", action="store_true")
    commands.add_argument("--local-dir", default="")
    table = sub.add_parser("model-bytes")
    table.add_argument("model")
    args = parser.parse_args(argv)
    if args.command == "worker-commands":
        config = load_config(args.config)
        local_dir = args.local_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "local-workers")
        for line in worker_commands(config, args.action, args.local, local_dir):
            print(line)
        return 0
    if args.command == "model-bytes":
        table = model_bytes(args.model)
        layer0 = table["layers"][0]
        print(json.dumps({"source": table["source"], "n_layer": table["n_layer"], "layer0": layer0,
                          "output": table["output"]}, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
