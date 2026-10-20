"""Identify the effective weights the TPU used for each 4/8-bit representation (single FC, down proj,
first 512 FFN columns, M=32). One-hot inputs e_k return column k of W_eff exactly; random inputs
measure end-to-end error. usage: int4_diag.py RUNS_ROOT WEIGHTS DIAG_DIR OUT.json"""
import json
import sys
from pathlib import Path

import numpy as np

import q4_0


def main():
    runs, weights, diag, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    L = q4_0.load(weights)
    C = 512
    q, d = L["down"][0][:, :C], L["down"][1][:, :C // 32]
    W = q4_0.dequantize(q, d)
    X = np.fromfile(diag / "rand_M32_K512.f32", "<f4").reshape(2, 32, C)
    cases = {"i4bw (BlockwiseQuantization, signed d)": ("diag", "diag-i4bw", None),
             "i4bw + DRQ flag": ("d4", "d4-i4bw-drq-onehot", "d4-i4bw-drq-rand"),
             "i4emu (emulated sub-channel pattern)": ("diag2", "diag-i4emu-onehot", "diag-i4emu-rand"),
             "i4pc + DRQ (per-channel int4 requant)": ("d4", "d4-i4pc-drq-onehot", "d4-i4pc-drq-rand"),
             "i8pc (per-channel int8, no DRQ flag)": ("d4", "d4-i8pc-plain-onehot", "d4-i8pc-plain-rand"),
             "i8pc + DRQ": ("d4", "d4-i8pc-drq-onehot", "d4-i8pc-drq-rand"),
             "f32 -> half": ("diag", "diag-f32", None)}
    res = {}
    for name, (sub, onehot, rand) in cases.items():
        Y = np.concatenate([np.fromfile(runs / sub / f"out.{onehot}.case{c}.f32", "<f4").reshape(32, 5120) for c in range(2)])
        We = Y.T.astype(np.float64)
        ref = W[:, :64].astype(np.float64)
        qq = q[:, :64].astype(np.float64)
        r = {"weff_rel_vs_d_times_q": float(np.linalg.norm(We - ref) / np.linalg.norm(ref)),
             "weff_rel_vs_q_only": float(np.linalg.norm(We - qq) / np.linalg.norm(qq))}
        if rand:
            rel = []
            for c in range(2):
                y = np.fromfile(runs / sub / f"out.{rand}.case{c}.f32", "<f4").reshape(32, 5120).astype(np.float64)
                rr = X[c].astype(np.float64) @ W.T.astype(np.float64)
                rel.append(float(np.linalg.norm(y - rr) / np.linalg.norm(rr)))
            r["random_input_rel_l2"] = max(rel)
        res[name] = r
        print(f"{name:42s} " + " ".join(f"{k}={v:.3e}" for k, v in r.items()))
    out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
