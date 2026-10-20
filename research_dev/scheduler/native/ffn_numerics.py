#!/usr/bin/env python3
"""Numerical rows for plain FFN calls of a phone worker serving quantized weights, against independent references.

An FFN call of owned layer l takes rows x (n_embd, already ffn-normed by the desktop) and returns
``W_down(l) (silu(W_gate(l) x) * W_up(l) x)`` over the requested columns. This tool connects to a running worker
over TCP (adb forward, the AOA bridge, or a host build), performs the HELLO, sends full-width calls with seeded
inputs at rows 1/2/4 for every requested layer, and compares each answer with two references computed in float64
numpy from the parent GGUF (the desktop's own artifact, sha256-pinned):

* ``ggml-cpu`` (the gate): what ggml's CPU kernels compute for Q4_0 / Q8_0 weights. Their vec-dot type is Q8_0,
  so each activation row is rounded to Q8_0 blocks of 32 (scale amax/127 stored as f16, values x / scale rounded
  half to even as the NEON and AVX quantizers do) before each product; block products are exact in float64
  (dequantized weight x dequantized activation), so the remaining difference is ggml's float32 accumulation order
  and the f16 wire.
* ``f32`` (reported): dequantized weights times the unrounded activations -- the size of the Q8_0 activation
  rounding itself, which every ggml CPU backend applies (the desktop's included).

The inputs are rounded to f16 exactly as the wire carries them, so both references see the worker's inputs.

    ffn_numerics.py --port 26991 --parent Qwen3-32B-Q8_0.gguf --artifact-sha256 sha256:2c50... \\
        --layers 21-40 --max-columns 25600 --rows 1,2,4 --out NUMERICS.json

PASS: every call's rel-L2 against the ``ggml-cpu`` reference is <= --threshold (default 1e-3) and every value is
finite. Schema ``s43-ffn-numerics-v1``.
"""

from __future__ import annotations

import argparse
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

SCHEMA = "s43-ffn-numerics-v1"
MAGIC = 0x46534631
VERSION = 6
HELLO_REQUEST = struct.Struct("<IHHQIIHH32s4x")
HELLO_RESPONSE = struct.Struct("<IHHHHIIIIII4xQQIHH32s")
EXECUTE_REQUEST = struct.Struct("<IHHIiIIIII")
EXECUTE_RESPONSE = struct.Struct("<IHHHHIiIIIIIQ")
MSG_HELLO, MSG_HELLO_RESPONSE = 1, 2
MSG_EXECUTE, MSG_EXECUTE_RESPONSE = 3, 4
FLAG_F16_IO, FLAG_SWIGLU = 1, 2
QK8_0 = 32
assert HELLO_REQUEST.size == 64 and HELLO_RESPONSE.size == 96
assert EXECUTE_REQUEST.size == 36 and EXECUTE_RESPONSE.size == 48


class FfnNumericsError(RuntimeError):
    pass


def fnv32(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def _receive(sock: socket.socket, size: int) -> bytes:
    chunks, remaining = [], size
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise FfnNumericsError("worker closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def q8_0_round_trip(x: np.ndarray) -> np.ndarray:
    """ggml quantize_row_q8_0 then dequantize, per row and block of 32: d = amax/127, q = x * (1/d) rounded half to
    even (computed with the unrounded d), value = q * f16(d). The SIMD quantizers the CPU backends run (NEON
    vcvtnq_s32_f32, plain and repacked; AVX _MM_ROUND_NEAREST) round half to even; only the scalar
    quantize_row_q8_0_ref uses roundf. f16 inputs hit exact ties often (2026-10-08, Qwen3-32B Q8_0 on the Pixel:
    every call with a tie that the two modes round apart read 0.9-2.9e-3 against a roundf reference, every call
    without one 2e-4)."""
    rows, width = x.shape
    if width % QK8_0:
        raise FfnNumericsError("activation width is not a multiple of 32")
    blocks = x.astype(np.float32).reshape(rows, width // QK8_0, QK8_0)
    amax = np.max(np.abs(blocks), axis=-1, keepdims=True)
    d = amax / np.float32(127.0)
    inverse = np.where(d > 0, np.float32(1.0) / np.where(d > 0, d, np.float32(1.0)), np.float32(0.0))
    q = np.rint(blocks * inverse)                                                # half to even, as the kernels do
    d16 = d.astype(np.float16).astype(np.float64)
    return (q.astype(np.float64) * d16).reshape(rows, width)


class ParentWeights:
    """One layer's FFN weights of the parent GGUF, dequantized to float64 (rows-major: shape (ne1, ne0))."""

    def __init__(self, path: Path):
        self.reader = gguf.GGUFReader(str(path))
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        self.types: dict[str, str] = {}

    def layer(self, layer: int) -> dict[str, np.ndarray]:
        result = {}
        for name in ("ffn_gate", "ffn_up", "ffn_down"):
            tensor = self.tensors[f"blk.{layer}.{name}.weight"]
            self.types[name] = tensor.tensor_type.name
            data = np.asarray(tensor.data)
            if tensor.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
                values = data.astype(np.float64)
            else:
                values = gguf.quants.dequantize(data, tensor.tensor_type).astype(np.float64)
            result[name] = values
        return result


def references(weights: dict[str, np.ndarray], x: np.ndarray) -> dict[str, np.ndarray]:
    """x: (tokens, n_embd) float32 as the worker received it -> {"ggml-cpu", "f32"} (tokens, n_embd) float64."""
    out = {}
    for mode in ("ggml-cpu", "f32"):
        rounded = q8_0_round_trip(x) if mode == "ggml-cpu" else x.astype(np.float64)
        gate = rounded @ weights["ffn_gate"].T
        up = rounded @ weights["ffn_up"].T
        # ggml computes the activation in float32
        gate32, up32 = gate.astype(np.float32), up.astype(np.float32)
        activation = (gate32 / (np.float32(1.0) + np.exp(-gate32))) * up32
        hidden = q8_0_round_trip(activation) if mode == "ggml-cpu" else activation.astype(np.float64)
        out[mode] = hidden @ weights["ffn_down"].T
    return out


def synthetic_inputs(rng: np.random.Generator, tokens: int, n_embd: int, outliers: int = 4) -> np.ndarray:
    """ffn-normed rows ~ N(0, 1) with a few large channels (normed decoder activations keep some)."""
    x = rng.normal(0.0, 1.0, size=(tokens, n_embd)).astype(np.float32)
    channels = rng.choice(n_embd, size=outliers, replace=False)
    x[:, channels] *= 20.0
    return x


class FfnSession:
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
            raise FfnNumericsError(f"HELLO refused: {reply[:4]}")
        self.hello = dict(n_embd=reply[5], n_ff=reply[6], offset=reply[7], max_columns=reply[8], weight_type=reply[9],
                          layer_count=reply[10], layer_mask=f"{reply[11]:016x}", weight_hash=f"{reply[12]:016x}",
                          column_quantum=reply[13], max_tokens=reply[14])
        self.next_id = 1

    def call(self, layer: int, x: np.ndarray, columns: int) -> tuple[np.ndarray, float, float]:
        tokens, n_embd = x.shape
        wire = x.astype(np.float16 if self.f16_io else np.float32).tobytes()
        request_id, self.next_id = self.next_id, self.next_id + 1
        started = time.perf_counter()
        self.sock.sendall(EXECUTE_REQUEST.pack(MAGIC, VERSION, MSG_EXECUTE, request_id, layer, x.size, len(wire),
                                               fnv32(wire), columns, tokens) + wire)
        header = EXECUTE_RESPONSE.unpack(_receive(self.sock, EXECUTE_RESPONSE.size))
        body = _receive(self.sock, header[8])
        rpc_ms = (time.perf_counter() - started) * 1000.0
        if (header[2] != MSG_EXECUTE_RESPONSE or header[3] != 0 or header[5] != request_id or header[6] != layer
                or header[7] != x.size or header[9] != fnv32(body) or header[11] != tokens):
            raise FfnNumericsError(f"execute response mismatch: {header}")
        values = np.frombuffer(body, dtype=np.float16 if self.f16_io else np.float32).astype(np.float32)
        return values.reshape(tokens, n_embd), rpc_ms, header[12] / 1000.0

    def close(self) -> None:
        self.sock.close()


def rel_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual.astype(np.float64) - expected) / max(np.linalg.norm(expected), 1e-30))


def run(args: argparse.Namespace) -> dict[str, object]:
    layers = parse_layer_spec(args.layers)
    rows_list = [int(value) for value in args.rows.split(",")]
    parent = ParentWeights(args.parent)
    session = FfnSession(args.host, args.port, artifact_sha256=args.artifact_sha256,
                         layers=layer_mask(parse_layer_spec(args.worker_layers or args.layers)),
                         n_embd=args.n_embd, max_columns=args.max_columns, max_tokens=args.max_tokens,
                         f16_io=not args.f32_io)
    rng = np.random.default_rng(args.seed)
    cases, worst, worst_f32 = [], 0.0, 0.0
    try:
        for layer in layers:
            weights = parent.layer(layer)
            for repeat in range(args.repeats):
                for rows in rows_list:
                    x = synthetic_inputs(rng, rows, args.n_embd)
                    if not args.f32_io:     # the wire carries f16: the references see the same rounded inputs
                        x = x.astype(np.float16).astype(np.float32)
                    actual, rpc_ms, compute_ms = session.call(layer, x, args.max_columns)
                    expected = references(weights, x)
                    gate, report = rel_l2(actual, expected["ggml-cpu"]), rel_l2(actual, expected["f32"])
                    worst, worst_f32 = max(worst, gate), max(worst_f32, report)
                    cases.append({"layer": layer, "repeat": repeat, "rows": rows, "rpc_ms": round(rpc_ms, 3),
                                  "compute_ms": round(compute_ms, 3), "rel_l2_ggml_cpu": gate, "rel_l2_f32": report,
                                  "finite": bool(np.isfinite(actual).all())})
            del weights
    finally:
        session.close()
    ok = bool(cases) and all(case["finite"] and case["rel_l2_ggml_cpu"] <= args.threshold for case in cases)
    return {"schema": SCHEMA, "status": "PASS" if ok else "FAIL", "threshold": args.threshold,
            "reference": "ggml-cpu (Q8_0 activation rounding, float64 products) from the parent GGUF",
            "worst_rel_l2": worst, "worst_rel_l2_f32_reference": worst_f32, "calls": len(cases), "layers": layers,
            "rows": rows_list, "io": "f32" if args.f32_io else "f16", "hello": session.hello,
            "weight_types": parent.types, "parent": str(args.parent), "artifact_sha256": args.artifact_sha256,
            "inputs": f"synthetic seed={args.seed}", "cases": cases}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent", type=Path, required=True, help="the parent GGUF (the desktop's artifact)")
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--layers", required=True, help="layers to call")
    parser.add_argument("--worker-layers", default=None, help="the worker's whole --layers spec (HELLO mask)")
    parser.add_argument("--rows", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--n-embd", type=int, default=5120)
    parser.add_argument("--max-columns", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--f32-io", action="store_true")
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (FfnNumericsError, OSError, KeyError, ValueError) as error:
        print(f"ffn_numerics: {error}", file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "worst_rel_l2", "worst_rel_l2_f32_reference", "calls",
                                                   "io")}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
