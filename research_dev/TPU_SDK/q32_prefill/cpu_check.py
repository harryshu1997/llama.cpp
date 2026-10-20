"""LiteRT CPU check of an exported (uncompiled) graph against the fp32 NumPy reference (smoke sizes)."""
import argparse
import json
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

import q4_0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("weights", type=Path)
    args = ap.parse_args()
    meta = json.loads((args.model_dir / "MODEL.json").read_text())
    M, C = meta["batch"], meta["columns"]
    L = q4_0.load(args.weights)
    w = {"gate": q4_0.dequantize(L["gate"][0][:C], L["gate"][1][:C]),
         "up": q4_0.dequantize(L["up"][0][:C], L["up"][1][:C]),
         "down": q4_0.dequantize(L["down"][0][:, :C], L["down"][1][:, :C // 32])}
    x = np.random.default_rng(7).standard_normal((M, 5120)).astype(np.float32) * 0.37
    g = x @ w["gate"].T
    ref = ((g / (1 + np.exp(-g))) * (x @ w["up"].T)) @ w["down"].T
    it = Interpreter(model_path=str(args.model_dir / "ffn.tflite"), num_threads=8)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x)
    it.invoke()
    y = it.get_tensor(it.get_output_details()[0]["index"])
    print(json.dumps(dict(model=str(args.model_dir), rel_l2=float(np.linalg.norm(y - ref) / np.linalg.norm(ref)))))


if __name__ == "__main__":
    main()
