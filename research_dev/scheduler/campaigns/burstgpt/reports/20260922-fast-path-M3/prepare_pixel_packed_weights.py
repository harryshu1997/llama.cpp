"""Copy original packed FFN tensors without quantization or dtype conversion."""

import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("f16_proxy", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    source = gguf.GGUFReader(args.model)
    proxy = {t.name: t for t in gguf.GGUFReader(args.f16_proxy).tensors}
    selected = {f"blk.{layer}.ffn_{op}.weight" for layer in range(18, 24) for op in ("gate", "up", "down")}
    path = args.output / "QWEN_PACKED.ffn.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen3")
    writer.add_name("Private Pixel original Q4_K/Q6_K FFN slice")
    writer.add_block_count(40)
    writer.add_embedding_length(5120)
    writer.add_feed_forward_length(17408)
    rows = []
    for tensor in source.tensors:
        if tensor.name not in selected:
            continue
        if tensor.tensor_type not in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K):
            raise ValueError("unexpected original weight type")
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        f16 = proxy[tensor.name]
        indices = np.unique(np.linspace(0, tensor.data.shape[0]-1, 32, dtype=int))
        decoded = gguf.quants.dequantize(tensor.data[indices], tensor.tensor_type).astype(np.float16)
        expected = f16.data[indices]
        exact = np.array_equal(decoded.view(np.uint16), expected.view(np.uint16))
        if not exact:
            raise ValueError(f"F16 proxy rounding mismatch: {tensor.name}")
        rows.append(dict(name=tensor.name, type=tensor.tensor_type.name, shape=tensor.shape.tolist(),
            bytes=tensor.n_bytes, original_payload_sha256=hashlib.sha256(tensor.data.tobytes()).hexdigest(),
            f16_sampled_rows=len(indices), f16_sampled_values=decoded.size, f16_samples_exact=exact))
    if len(rows) != 18:
        raise ValueError("missing FFN tensor")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    copied = {t.name: t for t in gguf.GGUFReader(path).tensors}
    for row in rows:
        if hashlib.sha256(copied[row["name"]].data.tobytes()).hexdigest() != row["original_payload_sha256"]:
            raise ValueError("packed tensor payload changed")
    manifest = dict(status="PASS", path=str(path), sha256=digest(path), bytes=path.stat().st_size,
        original=str(args.model), original_bytes=args.model.stat().st_size, original_mtime_ns=args.model.stat().st_mtime_ns,
        f16_proxy=str(args.f16_proxy), tensors=rows, weight_copy="all tensor bytes exact",
        f16_rounding_check="32 sampled complete rows per tensor; not an exhaustive F16 comparison")
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
