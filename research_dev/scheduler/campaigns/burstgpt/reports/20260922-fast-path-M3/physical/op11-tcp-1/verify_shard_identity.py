"""Verify the existing full-column OP11 shard without changing its manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from qualify_op11_tcp import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.source / "gguf-py"))
    import gguf

    index = json.loads(args.index.read_text())
    if len(index["shards"]) != 1:
        raise ValueError("expected one OP11 shard")
    manifest = index["shards"][0]
    if manifest["column_offset"] != 0 or manifest["columns"] != index["n_ff"]:
        raise ValueError("this check requires all parent FFN columns")
    parent_hash = "sha256:" + digest(args.parent)
    shard_path = args.index.parent / manifest["path"]
    shard_hash = "sha256:" + digest(shard_path)
    parent = gguf.GGUFReader(str(args.parent))
    shard = gguf.GGUFReader(str(shard_path))
    parent_tensors = {tensor.name: tensor for tensor in parent.tensors}
    expected_names = {row["name"] for row in manifest["tensors"]}
    if {tensor.name for tensor in shard.tensors} != expected_names:
        raise ValueError("shard tensor names differ from manifest")
    comparisons = []
    for tensor in shard.tensors:
        source = parent_tensors[tensor.name]
        parent_tensor_hash = hashlib.sha256(source.data).hexdigest()
        shard_tensor_hash = hashlib.sha256(tensor.data).hexdigest()
        comparisons.append({
            "name": tensor.name, "bytes": int(tensor.data.nbytes),
            "shape": tensor.shape.tolist(), "parent_tensor_sha256": parent_tensor_hash,
            "shard_tensor_sha256": shard_tensor_hash,
            "match": bool(parent_tensor_hash == shard_tensor_hash
                          and source.shape.tolist() == tensor.shape.tolist()
                          and source.tensor_type == tensor.tensor_type)})
    passed = (parent_hash == index["parent_sha256"] == manifest["parent_sha256"]
              and shard_hash == manifest["shard_sha256"]
              and all(row["match"] for row in comparisons))
    result = {"status": "PASS" if passed else "FAIL", "parent_sha256": parent_hash,
              "shard_sha256": shard_hash, "original_index_parent_verified": index["parent_verified"],
              "index_unmodified": True, "tensors": comparisons,
              "matched_tensors": sum(row["match"] for row in comparisons),
              "finished_epoch_s": time.time()}
    save(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "tensors"}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
