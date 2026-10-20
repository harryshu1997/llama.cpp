#!/usr/bin/env python3
"""Numerical rows for WS15 fused attention-projection calls, against an independent float32 reference.

A fused call of owned layer l takes rows [attention output (n_q) | residual (n_embd)] and returns
[delta (n_embd)] (``fused_execute_request``) or [delta | q | k | v] (``fused_next_execute_request``), with

    o      = W_o(l) attention
    x      = rms_norm(residual + o) * ffn_norm(l)
    delta  = o + W_down(l) (silu(W_gate(l) x) * W_up(l) x)
    y      = rms_norm(residual + delta) * attn_norm(l+1)
    q,k,v  = W_q(l+1) y, W_k(l+1) y, W_v(l+1) y        (raw: before q/k norm and RoPE)

This tool connects to a running worker over TCP (host build, or the Pixel through adb forward / the AOA bridge),
performs the normal HELLO plus the fused hello, sends fused calls with seeded inputs at rows 1/2/4, and compares
every returned segment with the reference computed in float32 numpy from the desktop's f16 proxy GGUF. The
inputs are rounded to f16 exactly as the wire carries them, so the reference sees the worker's inputs.

    fused_numerics.py --port 26991 --proxy Qwen3-14B-Q4KM-dequant-f16.gguf \\
        --artifact-sha256 sha256:d89e... --layers 1-3 --next-layers 1-3 --rows 1,2,4 --out NUMERICS.json

PASS: every segment of every call has rel-L2 <= --threshold (default 1e-3), and nothing is non-finite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "gguf-py") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "gguf-py"))
import gguf  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from ffn_shard_gguf import layer_mask, parse_layer_spec  # noqa: E402

SCHEMA = "s43-fused-numerics-v1"
MAGIC = 0x46534631
VERSION = 6
HELLO_REQUEST = struct.Struct("<IHHQIIHH32s4x")
HELLO_RESPONSE = struct.Struct("<IHHHHIIIIII4xQQIHH32s")
EXECUTE_REQUEST = struct.Struct("<IHHIiIIIII")
EXECUTE_RESPONSE = struct.Struct("<IHHHHIiIIIIIQ")
FUSED_HELLO = struct.Struct("<QQIIQfI")
MSG_HELLO, MSG_HELLO_RESPONSE = 1, 2
MSG_FUSED_HELLO, MSG_FUSED_HELLO_RESPONSE = 9, 10
MSG_FUSED, MSG_FUSED_NEXT, MSG_FUSED_RESPONSE = 11, 12, 13
FLAG_F16_IO, FLAG_SWIGLU = 1, 2
assert HELLO_REQUEST.size == 64 and HELLO_RESPONSE.size == 96
assert EXECUTE_REQUEST.size == 36 and EXECUTE_RESPONSE.size == 48 and FUSED_HELLO.size == 40


class FusedNumericsError(RuntimeError):
    pass


def fnv32(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def _receive(sock: socket.socket, size: int) -> bytes:
    chunks, remaining = [], size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise FusedNumericsError("worker closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ProxyWeights:
    """f16 proxy tensors, dequantized to float32 on demand (rows-major: shape (ne1, ne0))."""

    def __init__(self, path: Path):
        self.reader = gguf.GGUFReader(str(path))
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        architecture = self.reader.fields["general.architecture"].contents()
        self.rms_eps = float(self.reader.fields[f"{architecture}.attention.layer_norm_rms_epsilon"].contents())
        self._cache: dict[str, np.ndarray] = {}

    def get(self, name: str) -> np.ndarray:
        if name not in self._cache:
            tensor = self.tensors[name]
            data = np.asarray(tensor.data)
            if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                values = data.astype(np.float32)
            elif tensor.tensor_type == gguf.GGMLQuantizationType.F32:
                values = data.astype(np.float32)
            else:
                values = gguf.quants.dequantize(data, tensor.tensor_type).astype(np.float32)
            self._cache[name] = values
        return self._cache[name]


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps).astype(np.float32) * weight


def reference(weights: ProxyWeights, layer: int, attention: np.ndarray, residual: np.ndarray,
              next_qkv: bool) -> dict[str, np.ndarray]:
    """attention: (tokens, n_q), residual: (tokens, n_embd) -> segments, all float32 (tokens, width)."""
    w = lambda name: weights.get(name)  # noqa: E731
    p = f"blk.{layer}."
    o = attention @ w(p + "attn_output.weight").T
    x = rms_norm(residual + o, w(p + "ffn_norm.weight"), weights.rms_eps)
    gate = x @ w(p + "ffn_gate.weight").T
    up = x @ w(p + "ffn_up.weight").T
    activation = (gate / (1.0 + np.exp(-gate))) * up
    delta = o + activation @ w(p + "ffn_down.weight").T
    result = {"delta": delta}
    if next_qkv:
        n = f"blk.{layer + 1}."
        y = rms_norm(residual + delta, w(n + "attn_norm.weight"), weights.rms_eps)
        result["q"] = y @ w(n + "attn_q.weight").T
        result["k"] = y @ w(n + "attn_k.weight").T
        result["v"] = y @ w(n + "attn_v.weight").T
    return result


def synthetic_inputs(rng: np.random.Generator, tokens: int, n_q: int, n_embd: int,
                     outliers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Attention ~ N(0, 0.3); residual ~ N(0, 1) with a few large channels (decoder residual streams have them)."""
    attention = rng.normal(0.0, 0.3, size=(tokens, n_q)).astype(np.float32)
    residual = rng.normal(0.0, 1.0, size=(tokens, n_embd)).astype(np.float32)
    channels = rng.choice(n_embd, size=outliers, replace=False)
    residual[:, channels] *= 60.0
    return attention, residual


class FusedSession:
    def __init__(self, host: str, port: int, *, artifact_sha256: str, layers: int, n_embd: int, max_columns: int,
                 max_tokens: int, f16_io: bool, timeout_s: float = 120.0):
        self.f16_io = f16_io
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        digest = bytes.fromhex(artifact_sha256.removeprefix("sha256:"))
        flags = (FLAG_F16_IO if f16_io else 0) | FLAG_SWIGLU
        self.sock.sendall(HELLO_REQUEST.pack(MAGIC, VERSION, MSG_HELLO, layers, n_embd, max_columns, flags,
                                             max_tokens, digest))
        reply = HELLO_RESPONSE.unpack(_receive(self.sock, HELLO_RESPONSE.size))
        if reply[0] != MAGIC or reply[2] != MSG_HELLO_RESPONSE or reply[3] != 0:
            raise FusedNumericsError(f"HELLO refused: {reply[:4]}")
        self.hello = dict(n_embd=reply[5], n_ff=reply[6], offset=reply[7], max_columns=reply[8], weight_type=reply[9],
                          layer_count=reply[10], layer_mask=f"{reply[11]:016x}", weight_hash=f"{reply[12]:016x}",
                          column_quantum=reply[13], max_tokens=reply[14])
        self.next_id = 1

    def fused_hello(self, fused_mask: int, next_mask: int) -> dict[str, object]:
        payload = FUSED_HELLO.pack(fused_mask, next_mask, 0, 0, 0, 0.0, 0)
        self.sock.sendall(EXECUTE_REQUEST.pack(MAGIC, VERSION, MSG_FUSED_HELLO, 0, -1, 0, len(payload), fnv32(payload),
                                               0, 0) + payload)
        header = EXECUTE_RESPONSE.unpack(_receive(self.sock, EXECUTE_RESPONSE.size))
        body = _receive(self.sock, header[8])
        if header[2] != MSG_FUSED_HELLO_RESPONSE or header[9] != fnv32(body):
            raise FusedNumericsError("fused hello response is malformed")
        offered = FUSED_HELLO.unpack(body)
        result = dict(status=header[3], fused_layer_mask=f"{offered[0]:016x}", next_qkv_mask=f"{offered[1]:016x}",
                      n_q=offered[2], n_kv=offered[3], fused_weight_hash=f"{offered[4]:016x}", rms_eps=offered[5])
        if header[3] != 0:
            raise FusedNumericsError(f"fused hello refused: {result}")
        return result

    def call(self, layer: int, attention: np.ndarray, residual: np.ndarray, next_qkv: bool, columns: int,
             out_row: int) -> tuple[np.ndarray, float, float]:
        tokens = attention.shape[0]
        rows = np.concatenate([attention, residual], axis=1)
        wire = rows.astype(np.float16 if self.f16_io else np.float32).tobytes()
        request_id, self.next_id = self.next_id, self.next_id + 1
        message = MSG_FUSED_NEXT if next_qkv else MSG_FUSED
        started = time.perf_counter()
        self.sock.sendall(EXECUTE_REQUEST.pack(MAGIC, VERSION, message, request_id, layer, rows.size, len(wire),
                                               fnv32(wire), columns, tokens) + wire)
        header = EXECUTE_RESPONSE.unpack(_receive(self.sock, EXECUTE_RESPONSE.size))
        body = _receive(self.sock, header[8])
        rpc_ms = (time.perf_counter() - started) * 1000.0
        if (header[2] != MSG_FUSED_RESPONSE or header[3] != 0 or header[5] != request_id or header[6] != layer or
                header[7] != out_row * tokens or header[9] != fnv32(body) or header[11] != tokens):
            raise FusedNumericsError(f"fused response mismatch: {header}")
        values = np.frombuffer(body, dtype=np.float16 if self.f16_io else np.float32).astype(np.float32)
        return values.reshape(tokens, out_row), rpc_ms, header[12] / 1000.0

    def close(self) -> None:
        self.sock.close()


def rel_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm((actual - expected).astype(np.float64)) / max(np.linalg.norm(expected.astype(np.float64)), 1e-30))


def run(args: argparse.Namespace) -> dict[str, object]:
    layers = parse_layer_spec(args.layers)
    next_layers = parse_layer_spec(args.next_layers) if args.next_layers else []
    rows_list = [int(value) for value in args.rows.split(",")]
    weights = ProxyWeights(args.proxy)
    session = FusedSession(args.host, args.port, artifact_sha256=args.artifact_sha256,
                           layers=layer_mask(parse_layer_spec(args.worker_layers or args.layers)),
                           n_embd=args.n_embd, max_columns=args.max_columns, max_tokens=args.max_tokens,
                           f16_io=not args.f32_io)
    fused = session.fused_hello(layer_mask(layers), layer_mask(next_layers) if next_layers else 0)
    n_q, n_kv, n_embd = fused["n_q"], fused["n_kv"], args.n_embd
    rng = np.random.default_rng(args.seed)
    cases, worst = [], 0.0
    for repeat in range(args.repeats):
        for layer in layers:
            for next_qkv in ([True, False] if layer in next_layers else [False]):
                for rows in rows_list:
                    attention, residual = synthetic_inputs(rng, rows, n_q, n_embd)
                    if not args.f32_io:     # the wire carries f16: the reference sees the same rounded inputs
                        attention = attention.astype(np.float16).astype(np.float32)
                        residual = residual.astype(np.float16).astype(np.float32)
                    out_row = n_embd + (n_q + 2 * n_kv if next_qkv else 0)
                    actual, rpc_ms, compute_ms = session.call(layer, attention, residual, next_qkv, args.max_columns,
                                                              out_row)
                    expected = reference(weights, layer, attention, residual, next_qkv)
                    segments, offset = {}, 0
                    for name, width in [("delta", n_embd)] + ([("q", n_q), ("k", n_kv), ("v", n_kv)] if next_qkv else []):
                        part = actual[:, offset:offset + width]
                        offset += width
                        segments[name] = {"rel_l2": rel_l2(part, expected[name]),
                                          "finite": bool(np.isfinite(part).all())}
                        worst = max(worst, segments[name]["rel_l2"])
                    cases.append({"repeat": repeat, "layer": layer, "next_qkv": next_qkv, "rows": rows,
                                  "rpc_ms": round(rpc_ms, 3), "compute_ms": round(compute_ms, 3), "segments": segments})
    session.close()
    ok = all(segment["finite"] and segment["rel_l2"] <= args.threshold
             for case in cases for segment in case["segments"].values())
    return {"schema": SCHEMA, "status": "PASS" if ok else "FAIL", "threshold": args.threshold,
            "worst_rel_l2": worst, "calls": len(cases), "layers": layers, "next_layers": next_layers,
            "rows": rows_list, "io": "f32" if args.f32_io else "f16", "hello": session.hello, "fused_hello": fused,
            "proxy": str(args.proxy), "proxy_sha256": args.proxy_sha256, "inputs": f"synthetic seed={args.seed}",
            "cases": cases}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--proxy", type=Path, required=True, help="the desktop's f16 proxy GGUF (reference weights)")
    parser.add_argument("--proxy-sha256", default=None)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--layers", required=True, help="fused layers to call")
    parser.add_argument("--next-layers", default="", help="fused layers to also call with next Q/K/V")
    parser.add_argument("--worker-layers", default=None, help="the worker's whole --layers spec (HELLO mask)")
    parser.add_argument("--rows", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--n-embd", type=int, default=5120)
    parser.add_argument("--max-columns", type=int, default=17408)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--f32-io", action="store_true")
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (FusedNumericsError, OSError, KeyError, ValueError) as error:
        print(f"fused_numerics: {error}", file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "worst_rel_l2", "calls", "io")}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
