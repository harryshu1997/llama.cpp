"""Exhaustively compare a packed Pixel slice with its desktop F16 parent."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import gguf
import numpy as np


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return "sha256:" + result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    packed = gguf.GGUFReader(args.packed)
    parent = {tensor.name: tensor for tensor in gguf.GGUFReader(args.parent).tensors}
    expected = {f"blk.{layer}.ffn_{op}.weight"
                for layer in range(18, 24) for op in ("gate", "up", "down")}
    if {tensor.name for tensor in packed.tensors} != expected:
        raise ValueError("packed tensor set differs")
    rows = []
    for tensor in packed.tensors:
        target = parent[tensor.name]
        if target.tensor_type != gguf.GGMLQuantizationType.F16 or not np.array_equal(tensor.shape, target.shape):
            raise ValueError(f"parent geometry/type differs: {tensor.name}")
        mismatches = 0
        for start in range(0, tensor.data.shape[0], 64):
            decoded = gguf.quants.dequantize(tensor.data[start:start + 64], tensor.tensor_type).astype(np.float16)
            mismatches += int(np.count_nonzero(decoded.view(np.uint16) != target.data[start:start + 64].view(np.uint16)))
        row = {"name": tensor.name, "packed_type": tensor.tensor_type.name,
               "values": int(target.data.size), "mismatches": mismatches,
               "packed_tensor_sha256": hashlib.sha256(tensor.data).hexdigest()}
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {"status": "PASS" if all(row["mismatches"] == 0 for row in rows) else "FAIL",
              "parent": str(args.parent), "parent_sha256": digest(args.parent),
              "packed": str(args.packed), "packed_sha256": digest(args.packed),
              "tensors": rows, "finished_epoch_s": time.time(),
              "scope": "All stored weights after dequantization and F16 rounding; execution arithmetic is separately qualified."}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
