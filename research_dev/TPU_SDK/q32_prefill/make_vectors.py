"""Deterministic synthetic FFN inputs and the fp32 NumPy reference for one Qwen3-32B Q4_0 FFN layer.

x = rmsnorm(h) * ffn_norm.weight (layer's real norm weight), h = N(0,1) * per-channel lognormal(0, 0.5)
scale with 4 fixed outlier channels (x20). Each row of h has unit RMS after normalisation.
Reference: fp32 NumPy with the exactly dequantized Q4_0 weights:
  y = (silu(x @ gate.T) * (x @ up.T)) @ down.T
Writes per M: inputs_M{M}.f32 ([cases, M, 5120]), ref_M{M}.f32 ([cases, M, 5120]).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import q4_0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", type=Path, help="raw Q4_0 bytes of gate|up|down")
    ap.add_argument("norm", type=Path, help="ffn_norm.weight f32 [5120]")
    ap.add_argument("out", type=Path)
    ap.add_argument("--batches", default="1,32,128,512")
    ap.add_argument("--cases", type=int, default=2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    layer = q4_0.load(args.weights)
    w = {k: q4_0.dequantize(*v) for k, v in layer.items()}
    norm = np.fromfile(args.norm, dtype="<f4")
    rng = np.random.default_rng(20261008)
    chan = np.exp(rng.normal(0.0, 0.5, size=5120)).astype(np.float32)
    outliers = rng.choice(5120, size=4, replace=False)
    chan[outliers] *= 20.0
    record = {"weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "norm_sha256": hashlib.sha256(args.norm.read_bytes()).hexdigest(),
              "outlier_channels": outliers.tolist(), "batches": {}}
    for m in [int(b) for b in args.batches.split(",")]:
        xs, ys = [], []
        for c in range(args.cases):
            r = np.random.default_rng(1000 * m + c)
            h = r.standard_normal((m, 5120)).astype(np.float32) * chan
            h /= np.sqrt((h.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-6).astype(np.float32)
            x = (h * norm).astype(np.float32)
            g = x @ w["gate"].T
            u = x @ w["up"].T
            y = ((g / (1.0 + np.exp(-g))) * u) @ w["down"].T
            xs.append(x)
            ys.append(y.astype(np.float32))
        xs, ys = np.stack(xs), np.stack(ys)
        xs.astype("<f4").tofile(args.out / f"inputs_M{m}.f32")
        ys.astype("<f4").tofile(args.out / f"ref_M{m}.f32")
        # fp64 cross-check of the fp32 reference on the first case
        x64 = xs[0].astype(np.float64)
        g64 = x64 @ w["gate"].T.astype(np.float64)
        u64 = x64 @ w["up"].T.astype(np.float64)
        y64 = ((g64 / (1.0 + np.exp(-g64))) * u64) @ w["down"].T.astype(np.float64)
        rel = float(np.linalg.norm(ys[0] - y64) / np.linalg.norm(y64))
        record["batches"][m] = {"x_rms": float(np.sqrt((xs.astype(np.float64) ** 2).mean())),
                                "y_rms": float(np.sqrt((ys.astype(np.float64) ** 2).mean())),
                                "ref_fp32_vs_fp64_rel_l2": rel}
        print(m, record["batches"][m], flush=True)
    (args.out / "VECTORS.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
