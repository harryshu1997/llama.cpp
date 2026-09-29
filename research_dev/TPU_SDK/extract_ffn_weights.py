"""Extract one real F16 Qwen FFN slice without changing the source GGUF."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reader = GGUFReader(str(args.model))
    tensors = {t.name: t for t in reader.tensors}
    weights = {}
    for name in ("gate", "up", "down"):
        tensor = tensors[f"blk.{args.layer}.ffn_{name}.weight"]
        if tensor.data.dtype != np.float16 or not 0 < args.width <= 17408:
            raise ValueError("expected Qwen F16 source and a valid slice width")
        weights[name] = np.ascontiguousarray(tensor.data[:, -args.width:] if name == "down"
                                           else tensor.data[-args.width:, :])
    dest = args.output / "weights.npz"
    np.savez(dest, **weights)
    with args.model.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with dest.open("rb") as stream:
        output_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    record = dict(source=str(args.model), source_sha256=source_hash, layer=args.layer,
                  column_start=17408-args.width, width=args.width, embedding=5120,
                  weights_sha256=output_hash, shapes={k: list(v.shape) for k, v in weights.items()},
                  dtype="float16", extraction_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "WEIGHTS.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
