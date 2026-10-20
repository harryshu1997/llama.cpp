"""Exact GGML Q4_0 decoding for one Qwen3-32B FFN layer (gate, up, down).

Input: the raw byte range of blk.L.ffn_{gate,up,down}.weight copied verbatim from the GGUF
(three contiguous Q4_0 tensors, gate then up then down, as stored in the S42 FFN shard).
Q4_0 block (18 bytes): f16 d, 16 bytes qs; element j (0..15) = low nibble of qs[j],
element j+16 = high nibble; weight = d * (nibble - 8). d * q is exact in fp32.
"""
import numpy as np

N_EMBD, N_FF, QK = 5120, 25600, 32
SHAPES = {"gate": (N_FF, N_EMBD), "up": (N_FF, N_EMBD), "down": (N_EMBD, N_FF)}  # [out, in]
ORDER = ("gate", "up", "down")
BYTES = N_FF * N_EMBD // QK * 18


def decode(raw: bytes):
    """Return {name: (q int8 [out,in] in [-8,7], d float16 [out, in/32])}."""
    if len(raw) != 3 * BYTES:
        raise ValueError(f"expected {3 * BYTES} bytes, got {len(raw)}")
    out = {}
    for i, name in enumerate(ORDER):
        rows, cols = SHAPES[name]
        blocks = np.frombuffer(raw, dtype=np.uint8, count=BYTES, offset=i * BYTES).reshape(rows, cols // QK, 18)
        d = blocks[:, :, :2].copy().view("<f2").reshape(rows, cols // QK)
        qs = blocks[:, :, 2:]
        q = np.empty((rows, cols // QK, QK), dtype=np.int8)
        q[:, :, :16] = (qs & 0x0F).astype(np.int8) - 8
        q[:, :, 16:] = (qs >> 4).astype(np.int8) - 8
        out[name] = (q.reshape(rows, cols), d)
    return out


def dequantize(q, d):
    rows, cols = q.shape
    return (q.reshape(rows, cols // QK, QK).astype(np.float32) * d.astype(np.float32)[:, :, None]).reshape(rows, cols)


def load(path):
    with open(path, "rb") as f:
        return decode(f.read())
